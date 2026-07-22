"""
Runs the CPU-runnable Python blocks from
content/08-agents-harness/01-tool-use-function-calling.md, concatenated in
order so that later blocks can rely on names defined by earlier ones (as they
do in the chapter itself). Each tested block is copied verbatim from the
chapter; minimal glue needed to make it actually execute is added and clearly
marked "GLUE".

Blocks in the chapter (11 total, indices as given by the harness):
  #0  (line ~31)  JSON tool-schema example                  -- non-python
  #1  (line ~77)  Jinja chat-template fragment               -- non-python
  #2  (line ~103) training-data conversation fragment        -- non-python
  #3  (line ~142) parse-validate-dispatch pipeline            -- TESTED
  #4  (line ~272) full tool-call loop (uses `openai.OpenAI`)  -- SKIP(network)
  #5  (line ~490) worked-example JSON snippet (turn 1)        -- non-python
  #6  (line ~496) worked-example JSON snippet (turn 2)        -- non-python
  #7  (line ~512) execute_tool_calls_parallel()               -- TESTED
  #8  (line ~571) structured outputs (uses `openai.OpenAI`)   -- SKIP(network)
  #9  (line ~613) safe_dispatch() error-handling pattern      -- TESTED
  #10 (line ~687) build_tool_call_example() SFT-data builder  -- SKIP(network)*

* Block #10 is pure stdlib `json` logic with no actual network call in it;
  the heuristic classifier flagged it "needs-net" (likely because of the
  nearby prose mentioning "GPT-4, Claude 3.5" as data-generation sources).
  The task explicitly scoped this test to blocks #3, #7, #9 only, so block
  #10 is left out per that scope rather than re-litigating the classifier.

SKIP(network): blocks #4 and #8 both construct `openai.OpenAI(...)` clients
and call `client.chat.completions.create(...)` / `client.beta.chat.
completions.parse(...)` against the real OpenAI API. That is a live network
call requiring an API key -- forbidden in this harness (CI has no network/
keys). Rather than mock the entire OpenAI SDK surface to "run" a block whose
entire point is the live round trip, they are left defined-not-called.
However, blocks #7 and #9 both depend on `TOOL_FN_MAP`, which block #4 also
defines alongside the network-calling `run_tool_loop()`. `TOOL_FN_MAP` and
its two tool implementations (`_get_current_weather`, `_calculate`) contain
no network calls at all (they are local Python stubs), so that sub-slice of
block #4 is copied verbatim as GLUE -- it is a hard, network-free dependency
of the two tested blocks, not a rewrite of any tested logic.

The `jsonschema` package (used by block #3) is a third-party dependency not
on the guaranteed-available list, so its import is guarded; the
`jsonschema.validate(...)` calls only run if it is importable, and are
skipped (clearly labeled) otherwise.
"""

import concurrent.futures
import json
import re
from types import SimpleNamespace
from typing import Any

try:
    import jsonschema
except Exception:
    jsonschema = None


print("=" * 70)
print("Block #3 (line ~142): parse-validate-dispatch pipeline")
print("=" * 70)

# --- verbatim from the chapter (import jsonschema hoisted to guarded ---
# --- import above; rest is unchanged) -----------------------------------

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
# --- end verbatim ---

# Exercise the pipeline with a synthetic model output containing one
# well-formed tool call, exactly as the harness would receive it.
raw_model_output = (
    "Let me check that for you.\n"
    '<tool_call>{"name": "get_current_weather", '
    '"arguments": {"location": "Tokyo, Japan", "units": "celsius"}}</tool_call>'
)

# extract_tool_calls has no jsonschema dependency -- always exercised.
parsed_calls = extract_tool_calls(raw_model_output)
assert parsed_calls == [
    {"name": "get_current_weather",
     "arguments": {"location": "Tokyo, Japan", "units": "celsius"}}
]
print("extract_tool_calls ->", parsed_calls)

# extract_tool_calls must raise ValueError on malformed JSON (book-stated
# behavior, not a "deliberately broken" demo -- this is the documented
# error path of the function itself).
try:
    extract_tool_calls('<tool_call>{"name": "x", bad json}</tool_call>')
    raise AssertionError("expected ValueError for malformed tool-call JSON")
except ValueError as exc:
    print("extract_tool_calls correctly raised ValueError:", exc)

if jsonschema is not None:
    results = run_tool_call(raw_model_output)
    assert results == [{
        "name": "get_current_weather",
        "result": {"location": "Tokyo, Japan", "temperature": 22,
                    "units": "celsius", "conditions": "partly cloudy"},
    }]
    print("run_tool_call ->", results)

    # validate_tool_call must reject a call missing the required 'location'.
    try:
        validate_tool_call({"name": "get_current_weather", "arguments": {}})
        raise AssertionError("expected ValidationError for missing 'location'")
    except jsonschema.ValidationError as exc:
        print("validate_tool_call correctly rejected missing field:",
              exc.message)

    # validate_tool_call must reject an unknown tool name.
    try:
        validate_tool_call({"name": "get_stocks", "arguments": {}})
        raise AssertionError("expected ValueError for unknown tool")
    except ValueError as exc:
        print("validate_tool_call correctly rejected unknown tool:", exc)
else:
    print("SKIP(dependency): jsonschema not importable in this environment; "
          "validate_tool_call/run_tool_call (which call jsonschema.validate) "
          "were not exercised. extract_tool_calls was still fully exercised "
          "above since it has no jsonschema dependency.")


print()
print("=" * 70)
print("GLUE: network-free subset of block #4 (line ~272) needed by #7/#9")
print("=" * 70)
print("Only TOOL_FN_MAP and its two local stub implementations are copied "
      "verbatim; run_tool_loop() (the OpenAI-calling function) and the "
      "openai import are intentionally omitted -- see module docstring.")

# --- verbatim subset of block #4 ---
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
# --- end verbatim subset ---

# GLUE: a tiny fixture standing in for the OpenAI SDK's `ToolCall`/`Function`
# objects (the book's own inline comment in block #7 shows this exact shape:
# `ToolCall(id=..., function=Function(name=..., arguments=...))`). We don't
# import the real `openai` types since that would pull in the network-only
# package; a SimpleNamespace with the same `.id` / `.function.name` /
# `.function.arguments` attribute access is a faithful, minimal stand-in.
def _make_tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


print()
print("=" * 70)
print("Block #7 (line ~512): execute_tool_calls_parallel()")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

# Mirrors the book's own worked comment example: two independent
# get_current_weather calls for different cities, executed concurrently.
parallel_calls = [
    _make_tool_call("call_xyz", "get_current_weather", {"location": "London, UK"}),
    _make_tool_call("call_abc", "get_current_weather", {"location": "Tokyo, Japan"}),
]
parallel_results = execute_tool_calls_parallel(parallel_calls)
print("execute_tool_calls_parallel ->", parallel_results)

# Deterministically ordered by tool_call_id ("call_abc" < "call_xyz").
assert [r["tool_call_id"] for r in parallel_results] == ["call_abc", "call_xyz"]
tokyo_result = json.loads(parallel_results[0]["content"])
london_result = json.loads(parallel_results[1]["content"])
assert tokyo_result == {"location": "Tokyo, Japan", "temperature": 28,
                         "units": "celsius", "conditions": "sunny"}
assert london_result == {"location": "London, UK", "temperature": 14,
                          "units": "celsius", "conditions": "cloudy"}


print()
print("=" * 70)
print("Block #9 (line ~613): safe_dispatch() error-handling pattern")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

# Happy path: valid call executes and returns the tool's raw result.
ok_call = _make_tool_call("call_1", "calculate", {"expression": "14 * 9/5 + 32"})
ok_result = safe_dispatch(ok_call)
print("safe_dispatch (happy path) ->", ok_result)
assert ok_result == {"expression": "14 * 9/5 + 32", "result": 57.2}

# Step 1 error path: malformed JSON in the arguments string.
bad_json_call = SimpleNamespace(
    id="call_2",
    function=SimpleNamespace(name="calculate", arguments="{not valid json"),
)
bad_json_result = safe_dispatch(bad_json_call)
print("safe_dispatch (invalid json) ->", bad_json_result)
assert bad_json_result["error"] == "invalid_json"

# Step 2 error path: unknown tool name.
unknown_call = _make_tool_call("call_3", "get_stocks", {"ticker": "MSFT"})
unknown_result = safe_dispatch(unknown_call)
print("safe_dispatch (unknown tool) ->", unknown_result)
assert unknown_result == {
    "error": "unknown_tool",
    "message": "Tool 'get_stocks' does not exist.",
    "available_tools": list(TOOL_FN_MAP.keys()),
}

# Step 3 error path: wrong argument name triggers a TypeError inside the
# tool call, which safe_dispatch converts to a "bad_arguments" result.
bad_args_call = _make_tool_call("call_4", "calculate", {"typo_expr": "1+1"})
bad_args_result = safe_dispatch(bad_args_call)
print("safe_dispatch (bad arguments) ->", bad_args_result)
assert bad_args_result["error"] == "bad_arguments"


print()
print("=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)
