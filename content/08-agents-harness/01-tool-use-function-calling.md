# 8.1 Tool Use & Function Calling

A language model that can only read and write text is like a calculator without the equals key — impressive internally but frustratingly incomplete for real tasks. Tool use, also called *function calling*, is the mechanism that lets a model reach outside its context window and invoke the world: run code, query a database, look up today's weather, book a calendar event, or call any API imaginable. It is the foundational primitive on top of which every agent framework — ReAct, plan-and-execute, coding agents, multi-agent orchestrators — is built.

This chapter covers the full engineering picture: how tool schemas are defined, how the function-calling fine-tune works, how you parse and validate model outputs, how to run the tool-call loop, how to handle errors gracefully, and how structured outputs relate to function calling. We also look at parallel tool calls, streaming, and best practices for production agents. By the end, you will be able to build a correct, robust tool-calling loop from scratch and understand what is happening inside the model when it decides to call a function.

This chapter is the foundation for [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html), [Harness Engineering: Building a Coding Agent](../08-agents-harness/03-harness-coding-agent.html), and [Memory Systems for Agents](../08-agents-harness/05-agent-memory.html).

---

## Why Models Need Tools

Language models are frozen at training time. Their parametric knowledge is a lossy compression of their training corpus, not a live index of the world. Without tools, every answer about current events, exact arithmetic, external databases, or user-specific state is either wrong, hallucinated, or stale.

Three fundamental limitations motivate tool use:

1. **Knowledge cutoff.** A model trained through date $T$ cannot know what happened after $T$. Connecting it to a search engine or a live API dissolves this limit.
2. **Exact computation.** Transformers are surprisingly poor at reliable arithmetic and symbolic manipulation (see [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html) for the mechanistic reason). Delegating computation to a Python interpreter or calculator produces exact results.
3. **Stateful action.** An agent that can only generate text cannot create a file, send an email, or call an API — actions that require side effects in external systems.

Tool use is *not* about making the model smarter; it is about augmenting a powerful reasoning engine with reliable, exact, up-to-date actuators.

---

## Tool Schemas: Defining the Interface

Before a model can call a tool, it needs a structured description of what the tool does, what arguments it accepts, and what it returns. This description is called a *tool schema* or *function schema*, and it is injected into the model's context — typically in the system prompt or a dedicated tools section of the chat template.

The de facto standard for tool schemas is a subset of JSON Schema. Here is a complete example schema for a weather tool:

```json
{
  "name": "get_current_weather",
  "description": "Retrieve the current weather for a given city. Returns temperature, conditions, humidity, and wind speed. Only use this when the user asks about current or near-future weather.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City and optionally country, e.g. 'San Francisco, CA' or 'Tokyo, Japan'."
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature unit. Defaults to celsius."
      }
    },
    "required": ["location"]
  }
}
```

Several elements are worth highlighting:

- **`description` on the tool itself.** This is the most important field. The model reads this to decide *whether* to call the tool at all. Vague descriptions produce erratic call behavior. Be explicit about when the tool should and should not be invoked.
- **`description` on each parameter.** Format hints, examples, and constraints all belong here. The model will try to satisfy them.
- **`required`** lists which parameters the model must provide. Parameters not listed are optional and may be omitted.
- **`enum`** constrains a parameter to a fixed set of values, which dramatically reduces malformed outputs.
- **`type`** maps to JSON Schema primitive types: `string`, `number`, `integer`, `boolean`, `array`, `object`.

### Schema best practices

| Practice | Why it matters |
|---|---|
| Keep descriptions action-oriented ("Retrieve X given Y") | Guides the model's call-or-not decision |
| List all enum values explicitly | Prevents the model from inventing invalid values |
| Use `additionalProperties: false` for nested objects | Prevents hallucinated extra fields |
| Add a `format` hint for strings (e.g., `"format": "date-time"`) | Improves conformance |
| Keep parameter names consistent with your codebase | Reduces translation bugs in harness code |

### How schemas reach the model

Different providers and open-source frameworks serialize the tool list differently. The OpenAI Chat Completions API puts tools in a top-level `tools` array. HuggingFace chat templates encode them as a system message using a Jinja template that calls `tools | tojson`. The model is fine-tuned to understand whichever serialization format it was trained on.

Here is a simplified view of how a Jinja chat template injects tools into the prompt:

```text
{%- if tools %}
<|im_start|>system
You have access to the following tools:
{{ tools | tojson(indent=2) }}

Call tools using <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
The result will be returned in <tool_response>...</tool_response>.
<|im_end|>
{%- endif %}
```

The exact XML-like or JSON-like wrapper tokens vary per model family. Llama 3.1 uses a `<|python_tag|>` prefix for code interpreter calls and a custom tool-call format. Mistral uses `[TOOL_CALLS]` markers. Claude uses a dedicated `<parameter name="name">` XML structure. What they all share is that the *schema is in the context* and the *call is in the generated text*.

{{fig:tool-call-anatomy}}

---

## The Function-Calling Fine-Tune

A vanilla pretrained model cannot reliably produce well-formed tool call syntax. The function-calling capability is taught during supervised fine-tuning (SFT) and sometimes reinforced with RL (see [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) for the SFT pipeline).

### Training data format

Training data for function calling consists of multi-turn conversations where tool calls and their results appear in specific positions. A typical training example looks like:

```text
[system]: You have tools: [{"name": "search", ...}]
[user]: What's the capital of France?
[assistant]: <tool_call>{"name": "search", "arguments": {"query": "capital of France"}}</tool_call>
[tool]: {"result": "Paris"}
[assistant]: The capital of France is Paris.
```

The model learns simultaneously:
1. When to call a tool (the call decision).
2. Which tool to call (tool selection).
3. How to form the argument JSON (argument generation).
4. How to synthesize a final answer from tool results (grounded response).

The training loss is computed only on the assistant tokens — the tool call JSON and the final response — not on the user turns or tool results (which are fixed ground truth).

### What the model actually learns

From a representation-learning perspective, the model learns to:
- Map natural-language intent (e.g., "what's the weather") onto a tool's `description` via approximate semantic matching.
- Generate a JSON object whose token distribution is conditioned on the schema (field names, types, constraints) that appears earlier in the context.
- Recognize when tool results are sufficient to answer without further calls.

The function-calling capability generalizes across tools never seen during training because the schema is in-context. The model does not memorize specific tool names; it learns the *meta-skill* of reading a schema and producing conformant output. This is the same mechanism discussed in [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) — the schema tokens attend to the generation tokens, providing strong conditioning.

### RL on top of SFT

OpenAI's original GPT-4 function calling and subsequent work augmented SFT data with RL signals: if a tool call resulted in a successful downstream task completion, the call was rewarded. This teaches the model to be more conservative about calling tools unnecessarily and more aggressive about calling them when needed. See [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html) for the general pipeline.

---

## Parsing and Validation

The model outputs a string. Your harness must parse that string into a structured call object, validate it against the schema, and dispatch to the right function. Getting this right is where most production bugs live.

### The parse-validate-dispatch pipeline

```python
import json
import re
import jsonschema
from typing import Any

# --- Tool registry ---------------------------------------------------------
# Maps tool name -> (python_callable, json_schema_for_parameters)
TOOL_REGISTRY: dict[str, tuple] = {}

def register_tool(name: str, schema: dict):
    """Decorator that registers a Python function as a callable tool."""
    def decorator(fn):
        TOOL_REGISTRY[name] = (fn, schema)
        return fn
    return decorator

# --- Schema definitions ----------------------------------------------------
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"],
    "additionalProperties": False
}

@register_tool("get_current_weather", WEATHER_SCHEMA)
def get_current_weather(location: str, units: str = "celsius") -> dict:
    """Stub: in production this calls a real weather API."""
    return {"location": location, "temperature": 22, "units": units,
            "conditions": "partly cloudy"}

# --- Parser ----------------------------------------------------------------
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)

def extract_tool_calls(text: str) -> list[dict]:
    """
    Extract zero or more tool call objects from model output text.
    Returns a list of dicts, each with keys 'name' and 'arguments'.
    Raises ValueError if JSON is malformed.
    """
    matches = TOOL_CALL_RE.findall(text)
    calls = []
    for raw_json in matches:
        try:
            call = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed tool call JSON: {exc}\nRaw: {raw_json}")
        if "name" not in call:
            raise ValueError(f"Tool call missing 'name' field: {call}")
        calls.append(call)
    return calls

# --- Validator -------------------------------------------------------------
def validate_tool_call(call: dict) -> None:
    """
    Validate that the tool exists in the registry and that its
    arguments conform to the registered JSON schema.
    Raises ValueError or jsonschema.ValidationError on failure.
    """
    name = call.get("name")
    if name not in TOOL_REGISTRY:
        available = list(TOOL_REGISTRY.keys())
        raise ValueError(f"Unknown tool '{name}'. Available: {available}")
    _, schema = TOOL_REGISTRY[name]
    args = call.get("arguments", {})
    # This raises jsonschema.ValidationError on schema mismatch
    jsonschema.validate(instance=args, schema=schema)

# --- Dispatcher ------------------------------------------------------------
def dispatch_tool_call(call: dict) -> Any:
    """
    Execute a validated tool call. Returns the raw Python return value,
    which the harness will serialize back to JSON for the model.
    """
    fn, _ = TOOL_REGISTRY[call["name"]]
    args = call.get("arguments", {})
    return fn(**args)  # keyword-unpack the arguments dict

# --- Top-level entry point -------------------------------------------------
def run_tool_call(raw_text: str) -> list[dict]:
    """
    Full pipeline: parse -> validate -> dispatch for all calls in raw_text.
    Returns a list of result dicts: [{"name": ..., "result": ...}, ...]
    """
    calls = extract_tool_calls(raw_text)
    results = []
    for call in calls:
        validate_tool_call(call)
        result = dispatch_tool_call(call)
        results.append({"name": call["name"], "result": result})
    return results
```

### Error taxonomy

Not all parse failures are equal. A robust harness distinguishes between:

| Error type | Example | Recovery strategy |
|---|---|---|
| JSON syntax error | Missing closing brace | Return error to model; ask it to retry |
| Unknown tool | Model hallucinated `get_stocks` | Return error listing valid tools |
| Missing required arg | `location` omitted | Return error naming the missing field |
| Wrong arg type | `units: 42` instead of string | Return schema fragment; ask model to fix |
| Tool execution error | Network timeout, API 500 | Return truncated error message; log for ops |

The key insight: **errors are tool results**. Feed them back into the conversation just like a successful result, and the model can usually correct itself on the next turn. Do not silently swallow errors — the model cannot fix what it cannot see.

!!! warning "Never trust the model's JSON blindly"
    Even a well fine-tuned model will occasionally emit JSON with trailing commas (invalid), single-quoted strings (invalid), or Python-style `True`/`None` booleans (invalid in JSON). Always parse with a strict parser like `json.loads`. Consider a lenient fallback like `json5` or a regex-clean pass for production systems where latency matters more than strictness.

---

## The Tool-Call Loop

A single tool call is rare. Real tasks require multiple calls, sometimes in sequence (where later calls depend on earlier results) and sometimes in parallel (where calls are independent). The orchestration of this sequence is called the *tool-call loop* or *agentic loop*.

### Architecture of the loop

{{fig:toolcall-loop-architecture}}

### Complete loop implementation

```python
import json
import os
from dataclasses import dataclass, field
from typing import Any
from openai import OpenAI  # pip install openai

# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------
@dataclass
class Message:
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list | None = None   # populated by assistant when calling tools
    tool_call_id: str | None = None  # populated for role="tool" responses
    name: str | None = None          # tool name for role="tool"

def message_to_api_dict(m: Message) -> dict:
    """Convert our Message dataclass to the OpenAI API dict format."""
    d: dict[str, Any] = {"role": m.role}
    if m.content is not None:
        d["content"] = m.content
    if m.tool_calls is not None:
        d["tool_calls"] = m.tool_calls
    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    if m.name is not None:
        d["name"] = m.name
    return d

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get the current weather in a city. Use this when the user "
                "asks about current weather or temperature."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'London, UK'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression using Python's eval(). "
                "Use for arithmetic, conversions, and simple algebra. "
                "Example expression: '(32 - 32) * 5/9'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python arithmetic expression."
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _get_current_weather(location: str, units: str = "celsius") -> dict:
    """Stub weather API. Replace with a real HTTP call in production."""
    # Fake data for illustration
    data = {
        "London, UK": {"temp_c": 14, "conditions": "cloudy"},
        "Tokyo, Japan": {"temp_c": 28, "conditions": "sunny"},
    }
    info = data.get(location, {"temp_c": 20, "conditions": "unknown"})
    temp = info["temp_c"]
    if units == "fahrenheit":
        temp = temp * 9/5 + 32
    return {"location": location, "temperature": temp, "units": units,
            "conditions": info["conditions"]}

def _calculate(expression: str) -> dict:
    """Safely evaluate a math expression. Uses a restricted eval."""
    allowed_names = {"__builtins__": {}}
    try:
        result = eval(expression, allowed_names)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}

TOOL_FN_MAP = {
    "get_current_weather": _get_current_weather,
    "calculate": _calculate,
}

# ---------------------------------------------------------------------------
# The tool-call loop
# ---------------------------------------------------------------------------
def run_tool_loop(
    user_message: str,
    system_prompt: str = "You are a helpful assistant.",
    max_iterations: int = 10,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Run the full tool-call loop for a single user turn.
    Returns the final assistant text response.

    The loop:
      1. Build messages list with system + user.
      2. Call the model. If it emits tool calls, execute them and append
         their results, then call the model again.
      3. Stop when the model produces a plain text response (no tool calls)
         or when max_iterations is exhausted.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for iteration in range(max_iterations):
        # --- Model call ---
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            # "auto" lets the model decide; "required" forces at least one call;
            # {"type": "function", "function": {"name": "..."}} forces a specific call.
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Append the assistant's raw response to the conversation history
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None,
        })

        # --- Check for tool calls ---
        if not msg.tool_calls:
            # Model produced a final text answer — we're done
            return msg.content or ""

        # --- Execute every tool call the model requested ---
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                # Feed the parse error back so the model can recover
                result = {"error": f"JSONDecodeError: {exc}"}
            else:
                if fn_name not in TOOL_FN_MAP:
                    result = {"error": f"Unknown tool '{fn_name}'"}
                else:
                    try:
                        result = TOOL_FN_MAP[fn_name](**fn_args)
                    except Exception as exc:
                        # Execution errors are also fed back to the model
                        result = {"error": str(exc)}

            # Append the tool result as a "tool" role message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,   # must match the call's id
                "name": fn_name,
                "content": json.dumps(result),
            })

    # Exhausted iterations without a clean stop — return partial content
    return f"[max_iterations={max_iterations} reached without final answer]"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = (
        "What's the weather in London right now? "
        "And if it's 14°C there, what is that in Fahrenheit?"
    )
    answer = run_tool_loop(query)
    print(answer)
    # Expected: something like
    # "The weather in London is currently 14°C (cloudy).
    #  14°C is 57.2°F."
```

!!! example "Worked example: two-call chain"
    Consider the query: *"What is the weather in London, and what is that temperature in Fahrenheit?"*

    The model makes **two sequential calls**:

    **Turn 1 — model calls `get_current_weather`:**
    ```json
    {"name": "get_current_weather", "arguments": {"location": "London, UK", "units": "celsius"}}
    ```
    Tool returns: `{"temperature": 14, "units": "celsius", "conditions": "cloudy"}`

    **Turn 2 — model calls `calculate`:**
    ```json
    {"name": "calculate", "arguments": {"expression": "14 * 9/5 + 32"}}
    ```
    Tool returns: `{"result": 57.2}`

    **Turn 3 — model produces final answer:**
    > "The weather in London is 14°C (cloudy). That is **57.2°F**."

    Total tokens for this 3-turn exchange on the order of 400–600 prompt tokens plus ~50 completion tokens — on the order of USD 0.001 with a small model. Context window growth is $O(n)$ in the number of tool calls because every call + result is appended to the message history.

---

## Parallel Tool Calls

When the model needs results from multiple independent tools, it can emit all calls in a single response. This is called *parallel tool calling* and it was introduced in the OpenAI API in late 2023. It halves latency for independent subtasks.

```python
# The model response.choices[0].message.tool_calls may contain multiple items:
#
# tool_calls = [
#   ToolCall(id="call_abc", function=Function(name="get_current_weather", arguments='{"location":"Tokyo"}')),
#   ToolCall(id="call_xyz", function=Function(name="get_current_weather", arguments='{"location":"London"}')),
# ]
#
# Execute them concurrently, then append BOTH results before the next LLM call.

import concurrent.futures

def execute_tool_calls_parallel(tool_calls: list) -> list[dict]:
    """
    Execute a list of tool calls concurrently using a thread pool.
    Returns a list of tool-result message dicts, one per call.
    """
    def run_one(tc):
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments)
            result = TOOL_FN_MAP[fn_name](**fn_args)
        except Exception as exc:
            result = {"error": str(exc)}
        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": fn_name,
            "content": json.dumps(result),
        }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all calls simultaneously
        futures = {executor.submit(run_one, tc): tc for tc in tool_calls}
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Sort by tool_call_id to produce a deterministic ordering
    results.sort(key=lambda m: m["tool_call_id"])
    return results
```

!!! warning "Ordering of tool result messages matters"
    The OpenAI API requires that tool result messages appear in the *same order* as their corresponding tool calls in the assistant message. If you execute calls in parallel and append results out of order, the API may reject the request or the model may correlate results to the wrong calls. Always sort results by `tool_call_id` before appending.

---

## Structured Outputs and JSON Mode

Function calling and *structured outputs* solve overlapping but distinct problems:

- **Function calling**: the model decides *whether* to call a tool and *which one*. The output is a tool invocation object, not prose.
- **Structured outputs / JSON mode**: the model is constrained to always produce valid JSON conforming to a schema, regardless of whether tools are involved. Useful for extraction, classification, and parsing tasks.

Both mechanisms use constrained decoding under the hood (see [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html) for the full theory). The key insight is that once you have a JSON Schema, a CFG (context-free grammar) can be derived that accepts exactly the set of strings conforming to the schema. Token sampling is then masked so only tokens that could continue a valid prefix are allowed.

The practical implication: when you set `response_format={"type": "json_schema", "json_schema": {...}}`, the model cannot produce invalid JSON. Field names are not masked (the model still generates them from its parameters), but structural elements — braces, commas, colons, value types — are enforced.

```python
# Using structured outputs for a classification task (no tool needed)
from pydantic import BaseModel
from openai import OpenAI

class SentimentResult(BaseModel):
    sentiment: str      # "positive" | "negative" | "neutral"
    confidence: float   # 0.0 to 1.0
    rationale: str

client = OpenAI()
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Classify the sentiment of the user's text."},
        {"role": "user", "content": "The product is okay but the shipping was terrible."}
    ],
    response_format=SentimentResult,   # pydantic model -> auto-generated JSON schema
)
result: SentimentResult = completion.choices[0].message.parsed
print(result.sentiment, result.confidence)   # e.g. "negative" 0.72
```

### When to use structured outputs vs. function calling

| Use structured outputs when... | Use function calling when... |
|---|---|
| You want a fixed JSON payload every time | The model should decide whether to call |
| Extraction or classification tasks | The model may need to call multiple tools |
| No external system needs to be invoked | Actions with real-world side effects |
| You want guaranteed-valid JSON | The tool has an explicit Python implementation |

In practice, most agentic systems use both: function calling for tool invocations and structured outputs for parsing the final answer into a machine-readable form.

---

## Error Handling and Robustness

A production tool-call loop must handle a wide variety of failure modes gracefully. The goal is *graceful degradation*: never crash, always return something useful to the user, and log failures for observability.

### The error-as-tool-result pattern

```python
# Canonical error feedback pattern
def safe_dispatch(tool_call) -> dict:
    """
    Execute a single tool call with comprehensive error handling.
    Always returns a dict suitable for json.dumps().
    """
    fn_name = tool_call.function.name

    # Step 1: Parse arguments
    try:
        fn_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        return {
            "error": "invalid_json",
            "message": str(exc),
            "hint": "Please emit valid JSON in the 'arguments' field."
        }

    # Step 2: Check tool exists
    if fn_name not in TOOL_FN_MAP:
        return {
            "error": "unknown_tool",
            "message": f"Tool '{fn_name}' does not exist.",
            "available_tools": list(TOOL_FN_MAP.keys())
        }

    # Step 3: Execute with timeout
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("Tool execution exceeded time limit.")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(10)  # 10 second limit
    try:
        result = TOOL_FN_MAP[fn_name](**fn_args)
        signal.alarm(0)  # cancel alarm
        return result
    except TimeoutError:
        return {"error": "timeout", "message": "Tool call timed out after 10 seconds."}
    except TypeError as exc:
        # Wrong argument names or types
        return {"error": "bad_arguments", "message": str(exc)}
    except Exception as exc:
        # Catch-all: log internally, return generic message
        import traceback
        print(f"[ERROR] Tool '{fn_name}' raised: {traceback.format_exc()}")
        return {"error": "execution_error", "message": "An internal error occurred."}
```

### Maximum call limits and infinite loop prevention

A model with a bug in its reasoning (or a pathologically adversarial user) can trigger an infinite loop where it keeps calling the same tool with slightly different arguments. Always enforce:

1. **`max_iterations`** — a hard cap on how many times the model can call tools in one turn.
2. **Per-tool call quotas** — optionally limit how many times a single tool can be called per loop.
3. **Context window budget** — check that appending one more tool result will not overflow the context window before making the next LLM call.

The context window budget is worth quantifying. If each tool result is ~200 tokens and a model has a 128k context window with 50k tokens already consumed by the system prompt and conversation, we have budget for roughly $(128{,}000 - 50{,}000) / 200 \approx 390$ tool result messages before truncation is required. In practice, keep the loop limit well under 20 to stay in the low-latency regime.

!!! interview "Interview Corner"
    **Q:** A model in your production tool-calling agent is repeatedly calling the same search tool without making progress. What are the possible root causes, and how would you fix each one?

    **A:** Root causes fall into three buckets. First, *the tool is returning unhelpful results*: the search query is too broad or the tool is returning the same cached page. Fix: return richer metadata with results (URL, snippet length, freshness) and instruct the model to vary the query. Second, *the model is stuck in a reasoning loop*: the system prompt or few-shot examples do not demonstrate how to synthesize partial information into a final answer. Fix: add an explicit instruction like "If after two searches you still lack a definitive answer, tell the user what you found and what remains uncertain." Third, *a lack of a hard iteration limit*: without `max_iterations`, the loop never exits. Fix: enforce a cap and surface a partial-answer message rather than an infinite spin. In all cases, structured logging of each (call, result) pair is essential for debugging.

---

## Training Models for Tool Use

If you are fine-tuning a model to use tools rather than relying on a pretrained capability, here is what the data preparation and training loop look like.

### Constructing training examples

```python
import json

def build_tool_call_example(
    user_query: str,
    tool_call: dict,          # {"name": ..., "arguments": {...}}
    tool_result: dict,        # what the tool actually returned
    final_answer: str,
    tools: list[dict],        # the available tool schemas
    system_prompt: str = "You are a helpful assistant with tool access."
) -> list[dict]:
    """
    Build a multi-turn conversation suitable for SFT on tool use.
    Returns a list of message dicts in OpenAI chat format.

    The training loss should be computed ONLY on:
      - The assistant's tool call content (turn 3)
      - The assistant's final answer (turn 5)
    Not on: system, user, or tool result messages.
    """
    tool_call_str = json.dumps({"name": tool_call["name"],
                                 "arguments": tool_call["arguments"]})
    return [
        # Turn 1: system prompt includes serialized tool schemas
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"Available tools:\n{json.dumps(tools, indent=2)}"
            )
        },
        # Turn 2: user query (no loss computed here)
        {"role": "user", "content": user_query},
        # Turn 3: assistant decides to call a tool  ← COMPUTE LOSS HERE
        {
            "role": "assistant",
            "content": f"<tool_call>{tool_call_str}</tool_call>"
        },
        # Turn 4: tool result (no loss computed here — this is fixed truth)
        {
            "role": "tool",
            "name": tool_call["name"],
            "content": json.dumps(tool_result)
        },
        # Turn 5: final assistant answer  ← COMPUTE LOSS HERE
        {"role": "assistant", "content": final_answer}
    ]

# Example
example = build_tool_call_example(
    user_query="What is 15% of 240?",
    tool_call={"name": "calculate", "arguments": {"expression": "0.15 * 240"}},
    tool_result={"result": 36.0},
    final_answer="15% of 240 is 36.",
    tools=[{
        "name": "calculate",
        "description": "Evaluate a Python arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }]
)

for msg in example:
    print(f"[{msg['role']}] {str(msg['content'])[:80]}")
```

### Data sources for tool-call training

1. **Synthetic generation**: Use a capable model (GPT-4, Claude 3.5) to generate (query, tool call, result, answer) tuples given tool schemas. This scales cheaply and can cover arbitrary tool combinations.
2. **Templated math/code tasks**: For calculator-style tools, many reasoning benchmarks (GSM8K, MATH) can be automatically converted: "call `calculate` with the expression, then state the answer."
3. **Human demonstrations**: The highest quality but most expensive. Necessary for tools with complex multi-call chains.
4. **Negative examples**: Include examples where the model correctly decides *not* to call a tool (the answer is in its parametric knowledge). Without these, over-calling becomes a problem.

A rough rule of thumb: the order of thousands of tool-use examples is sufficient to teach a model the meta-skill on top of a strong instruction-following base. The exact count depends heavily on base model quality; a strong SFT base like Mistral-7B-Instruct can generalize to new schemas with on the order of 1,000–5,000 tool-call examples.

---

## Key Concepts in Practice

### Streaming with tool calls

When using streaming mode (`stream=True`), tool call arguments arrive token by token. The complete argument JSON is not available until the stream for that tool call ends. Most SDKs accumulate the delta and surface a complete `ToolCall` object at the end of the stream. For the harness, the simplest approach is to collect the full stream, then parse tool calls from the accumulated message rather than trying to parse partial JSON mid-stream.

### Tool call IDs and multi-turn history

Each tool call has a unique `id` (e.g., `call_abc123`). The corresponding tool result message must include this `id` as `tool_call_id`. This correlation allows the model to match results to calls when parallel calls are made. Lose the ID and the API will reject the message sequence.

### Context management across long tool chains

In a long agentic session, the message history grows without bound. Strategies to manage this (covered in depth in [Context Engineering & Management](../08-agents-harness/04-context-engineering.html)):

- **Truncate old tool results**: keep only the most recent $k$ tool call/result pairs.
- **Summarize intermediate results**: after every $n$ tool calls, ask the model to produce a concise summary and replace the raw results with it.
- **Structured memory**: extract key facts from tool results and store them in a dedicated memory section (see [Memory Systems for Agents](../08-agents-harness/05-agent-memory.html)).

### The Model Context Protocol (MCP)

The Model Context Protocol (MCP), introduced by Anthropic in late 2024, is a standardized JSON-RPC-based protocol that lets any server expose tools, resources, and prompts to any compliant client. Instead of writing bespoke tool dispatch code per application, MCP provides a universal adapter: a tool server speaks MCP, the harness speaks MCP, and new tools can be plugged in without changing the client. See [The Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html) for a deep dive.

---

!!! key "Key Takeaways"
    - Tool schemas are JSON Schema objects injected into the model's context; the `description` field is the most important signal the model uses to decide whether and how to call a tool.
    - Function-calling capability is taught during SFT on multi-turn conversations that include tool calls and results; the model learns the meta-skill of reading any schema and producing conformant output.
    - The parse-validate-dispatch pipeline must handle JSON syntax errors, unknown tools, wrong argument types, and execution errors — all by feeding errors back to the model as tool results.
    - The tool-call loop appends every (call, result) pair to the message history and repeats until the model produces a plain-text response or a maximum iteration limit is reached.
    - Parallel tool calls allow the model to emit multiple independent calls in one response; results must be appended in the same order as the calls before the next LLM invocation.
    - Structured outputs (JSON Schema constrained decoding) and function calling are complementary: function calling governs when to invoke a tool; structured outputs ensure the response payload is machine-readable.
    - Always enforce a maximum call limit and a context window budget check to prevent infinite loops and context overflow.
    - Training for tool use requires both positive examples (correct calls) and negative examples (correct non-calls) to avoid over-calling; a few thousand high-quality examples suffice on top of a strong instruction-following base.
    - Tool use is the foundational primitive for all agent architectures; everything in the agentic loop builds on top of the mechanisms described in this chapter.

---

!!! sota "State of the Art & Resources (2026)"
    Tool use and function calling are now table-stakes capabilities for frontier LLMs, with every major provider offering native JSON-Schema-defined tool dispatch and constrained structured outputs. Research focus has shifted from teaching the basic meta-skill to multi-hop agentic planning, reliable error recovery, and standardized tool protocols such as MCP.

    **Foundational work**

    - [Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023)](https://arxiv.org/abs/2302.04761) — first demonstration that a model can self-supervisedly learn when and how to call external APIs.
    - [Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2023)](https://arxiv.org/abs/2210.03629) — interleaves chain-of-thought reasoning traces with tool actions; canonical formalization of the agentic tool loop.

    **Recent advances (2023–2026)**

    - [Qin et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs* (2023)](https://arxiv.org/abs/2307.16789) — large-scale SFT dataset and benchmark spanning 16k+ real REST APIs; introduced depth-first search–based decision tree for multi-step tool planning.
    - [Patil et al., *Gorilla: Large Language Model Connected with Massive APIs* (2023)](https://arxiv.org/abs/2305.15334) — studied and reduced API hallucination via retrieval-augmented training; spawned the Berkeley Function Calling Leaderboard.
    - [Patil et al., *Berkeley Function Calling Leaderboard (BFCL)* (2024–2026)](https://gorilla.cs.berkeley.edu/leaderboard.html) — live, continuously updated benchmark evaluating single-call, parallel, multi-turn, and agentic tool-call accuracy across all major models.

    **Open-source & tools**

    - [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) — ICLR 2024 spotlight; training data, ToolLLaMA model, and evaluation harness for 16k-API tool use.
    - [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) — Gorilla OpenFunctions models, BFCL evaluation code, and GoEx safe execution engine.
    - [modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol) — Anthropic's open MCP specification; the emerging standard for universal tool/resource interfaces between LLM clients and servers.

    **Go deeper**

    - [OpenAI, *Function Calling* (official docs)](https://developers.openai.com/api/docs/guides/function-calling) — canonical API reference that de facto standardized JSON Schema tool definitions, parallel calls, and `strict` structured-output mode.
    - [LangChain, *Tool Calling with LangChain* (2024)](https://www.langchain.com/blog/tool-calling-with-langchain) — practical guide to the unified `tool_calls` interface across OpenAI, Anthropic, and Gemini providers.
    - [Anthropic, *Introducing the Model Context Protocol* (2024)](https://www.anthropic.com/news/model-context-protocol) — announcement and motivation for MCP as a universal adapter replacing bespoke tool dispatch code.

## Further Reading

- **Toolformer** — Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools," 2023. The foundational paper demonstrating self-supervised tool-use learning.
- **ToolBench / ToolLLM** — Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs," 2023. Large-scale benchmark and dataset for tool use across thousands of APIs.
- **HuggingFace `chat_templates` documentation** — Practical reference for how tool schemas are injected into prompt templates for open-source models.
- **OpenAI Function Calling documentation** — The API reference that de facto standardized the JSON Schema tool definition format.
- **ReAct** — Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023. Interleaves reasoning traces and tool calls; the canonical formalization of the agent loop (see [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html)).
- **Model Context Protocol (MCP)** — Anthropic, 2024. Open specification for a universal tool/resource interface; see [The Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html).
- **Gorilla** — Patil et al., "Gorilla: Large Language Model Connected with Massive APIs," 2023. Studied API hallucination and trained models to call APIs correctly from documentation.
