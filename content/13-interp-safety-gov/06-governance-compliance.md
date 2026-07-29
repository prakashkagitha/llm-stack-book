# 13.6 AI Governance, Compliance & Regulation

Shipping a large language model (LLM) product is no longer just a systems engineering problem; it is a legal and organisational engineering problem too. In the span of a few years, AI regulation has graduated from voluntary guidance documents to binding statutes with civil penalties, mandatory incident reporting, and third-party audits. Engineers who understand only the model stack — and not the compliance layer on top of it — will ship systems that expose their employers to nine-figure fines or force costly re-architectures after launch.

This chapter gives you the engineer's view of that compliance layer: the EU AI Act (the world's most comprehensive AI law), the NIST AI Risk Management Framework (AI RMF), ISO/IEC 42001, and the practical artefacts — model cards, datasheets, eval reports, audit trails — that satisfy all of them. We also cover the specific obligations that fall on frontier model providers, the systemic-risk threshold, copyright and provenance mechanics, and serious-incident reporting pipelines that you actually have to build.

Related background: [AI Safety: Scalable Oversight, Dangerous-Capability Evals & Frontier Safety](../13-interp-safety-gov/05-ai-safety-oversight.html) covers the technical safety work that feeds governance artefacts. [Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html) details the eval methods referenced in model card sections. [Watermarking, Provenance & AI-Content Detection](../13-interp-safety-gov/04-watermarking-provenance.html) covers technical provenance methods. [Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html) handles the data-subject rights angle.

---

## The Regulatory Landscape in 2025–2026

Before diving into specifics, a mental map helps. Several regulatory streams are converging simultaneously.


{{fig:gov-regulatory-landscape-map}}


These three frameworks are complementary: the EU AI Act tells you *what* you must do and by when; the NIST AI RMF tells you *how* to run the governance process; ISO/IEC 42001 tells you *how to prove* to an auditor that you run that process consistently.

The EU is the most prescriptive regime, which is why the bulk of this chapter is spent there, but it is not the only one an engineer will meet. In the **United States** there is still no comprehensive federal AI statute: Executive Order 14110 was rescinded in January 2025 and replaced by a deregulatory, competitiveness-focused federal posture (the 2025 "AI Action Plan"), which pushes the binding rules down to the states — Colorado's AI Act (SB 24-205), the first US state law placing duties on developers *and* deployers of "high-risk" AI, whose effective date has been postponed more than once, and California's SB 53 (Transparency in Frontier Artificial Intelligence Act, signed September 2025), which requires large frontier developers to publish a safety framework and report critical safety incidents to the state. The **UK** has taken a sectoral, regulator-led approach with no cross-cutting AI statute, with the AI Security Institute (renamed from AI Safety Institute in 2025) doing pre-deployment testing by agreement rather than by law. **China** has the most operationally intrusive regime for generative AI: the Interim Measures for Generative AI Services (2023), a filing/registration requirement for algorithms and large models, and labelling rules for AI-generated synthetic content in force since September 2025. Internationally, the Council of Europe Framework Convention on AI (opened for signature September 2024) and the G7 Hiroshima Process code of conduct are the main soft-law instruments. The practical consequence for a multinational deployment: build to the strictest applicable regime (usually the EU AI Act), then map the artefacts onto the others — the underlying evidence (model card, eval report, incident log, data-rights register) is largely the same.

---

## The EU AI Act: Structure and Timeline

The EU AI Act (Regulation (EU) 2024/1689) entered into force on 1 August 2024. Its obligations phase in over 36 months:

| Date | Obligation active |
|---|---|
| 2 Feb 2025 | Prohibited AI practices banned (Article 5); AI-literacy duty (Article 4) |
| 2 Aug 2025 | GPAI model obligations (Chapter V); governance bodies, notified bodies and penalties provisions |
| 2 Aug 2026 | High-risk application obligations (Annex III); Article 50 transparency duties; general applicability of the Act |
| 2 Aug 2027 | High-risk AI embedded in regulated products (Annex I); GPAI models already on the market before Aug 2025 must be brought into compliance |

Engineers need to care most about **2 Aug 2025** (GPAI model obligations — affects every frontier model provider) and **2 Aug 2026** (high-risk application rules — affects deployers building on those models).

!!! warning "The timetable is a moving target"

    Treat the table above as the position under the Regulation as enacted, not as gospel. In November 2025 the European Commission proposed a "Digital Omnibus" simplification package that would, among other things, postpone parts of the high-risk regime and adjust some GPAI and transparency provisions; that proposal had to pass the ordinary legislative procedure, so the dates that actually bind you may have shifted. Before you build a compliance plan around a date, check the current consolidated text on EUR-Lex and the European AI Office's guidance pages. Never hard-code a regulatory deadline into a design document without a dated citation next to it.

### Risk Tiers

The Act classifies AI systems into four tiers:

1. **Prohibited** — Social scoring by public authorities, real-time biometric surveillance in public spaces, AI that exploits vulnerable groups. No compliance path; simply illegal in the EU.
2. **High-risk** — Listed in Annex III (biometric identification, critical infrastructure, employment decisions, essential services, law enforcement, migration, administration of justice, democratic processes). Also anything that is a safety component of a product under EU product safety law.
3. **Limited-risk** — Chatbots, deep-fakes, emotion recognition. Transparency obligations only (must disclose AI-generated content to users).
4. **Minimal-risk** — Spam filters, AI in video games. No mandatory obligations.

Most LLM applications land in **limited-risk** if deployed for open consumer use, but many enterprise applications (hiring tools, medical triage, credit scoring) tip into **high-risk**, triggering a heavy compliance programme.

{{fig:gov-eu-ai-act-risk-pyramid}}

### Article 50: The Transparency Tier Has Teeth Too

"Limited risk" sounds like "nothing to do", but Article 50 (applicable from 2 Aug 2026) imposes concrete engineering work on almost every LLM product:

- **Disclose the machine.** Systems that interact directly with people must tell the person they are talking to an AI, unless it is obvious from context.
- **Mark synthetic output machine-readably.** Providers of systems that generate synthetic audio, image, video *or text* must ensure the outputs are marked in a machine-readable format and detectable as artificially generated or manipulated, with solutions that are "effective, interoperable, robust and reliable as far as this is technically feasible".
- **Label deep fakes and synthetic news text.** Deployers who publish deep fakes, or AI-generated text published to inform the public on matters of public interest, must disclose that fact.

The machine-readable marking duty is the one that forces a technical decision. In practice teams satisfy it with two complementary layers: **cryptographic content credentials** (the C2PA / Content Credentials standard, with the open-source `c2pa-rs` and `c2patool` implementations, which sign provenance manifests into media files) and **statistical watermarking** of generated text (SynthID-Text and the Kirchenbauer et al. green-list scheme are the reference approaches). Neither is a complete answer — signatures are stripped by re-encoding, text watermarks are degraded by paraphrase — which is exactly why the Act qualifies the duty with "as far as technically feasible". See [Watermarking, Provenance & AI-Content Detection](../13-interp-safety-gov/04-watermarking-provenance.html) for the mechanisms and their attack surface.

---

## General-Purpose AI (GPAI) Model Obligations

Chapter V of the Act (Articles 51–56, with the documentation contents spelled out in Annexes XI and XII) creates a distinct regime for "general-purpose AI models" (GPAI) — models trained on broad data at scale that can be adapted to many downstream tasks. Practically, this means every large pre-trained language model, including models released open-weight.

The key definitions:

- **GPAI model** — Any AI model trained with a large amount of data using self-supervision at scale, exhibiting significant generality and capable of performing a wide range of distinct tasks.
- **GPAI model with systemic risk** — A GPAI model whose training used compute exceeding $10^{25}$ floating-point operations (FLOPs). The Commission can adjust this threshold by delegated act.

!!! example "Worked Example: The 1e25-FLOP Threshold"

    The systemic-risk threshold is $C_{\text{train}} \geq 10^{25}$ FLOPs.

    For a dense transformer with $N$ parameters trained on $D$ tokens, the Chinchilla approximation gives:

    $$
    C_{\text{train}} \approx 6 \cdot N \cdot D
    $$

    A 70 B parameter model trained on 2 T tokens:

    $$
    C = 6 \times 7 \times 10^{10} \times 2 \times 10^{12}
      = 6 \times 1.4 \times 10^{23}
      = 8.4 \times 10^{23} \text{ FLOPs}
    $$

    This is about $10^{23.9}$, safely below the $10^{25}$ bar.

    A 400 B parameter model trained on 15 T tokens:

    $$
    C = 6 \times 4 \times 10^{11} \times 1.5 \times 10^{13}
      = 3.6 \times 10^{25} \text{ FLOPs}
    $$

    This *exceeds* $10^{25}$, triggering systemic-risk obligations. By 2026 this is no longer an exclusive club: essentially every frontier flagship model — successive releases from OpenAI, Google DeepMind, Anthropic, Meta, xAI, and DeepSeek — is trained well above the threshold, and even the 2023–2024 generation (GPT-4, Gemini Ultra, Claude 3 Opus, and Meta's Llama 3.1 405B) already sat in or near this territory. Most research-scale and smaller open-weight models still sit below it.

### GPAI Obligations for All Providers (Article 53)

Every GPAI model provider, regardless of compute, must:

1. **Technical documentation** — Maintain up-to-date documentation covering architecture, training data, compute, evaluation results, known limitations, intended and foreseeable uses, and content filtering measures.
2. **Training-data summary** — Publish a "sufficiently detailed summary" of training data used. The Office of AI (European AI Office) publishes a template; it includes: data sources, languages covered, data selection methodology, filtering applied, personal data handling, and copyright measures.
3. **Copyright compliance** — Implement a policy to comply with EU copyright law, including the text-and-data mining exceptions in the 2019 Copyright Directive. Retain records to demonstrate compliance.
4. **Downstream deployer information** — Provide AI system providers who integrate the GPAI model with documentation and instructions sufficient to comply with their own obligations.

Mapping to the letters of Article 53(1), which you will need when you cite them in a compliance document: (a) technical documentation for the AI Office and national authorities (contents in Annex XI); (b) documentation for downstream providers (contents in Annex XII); (c) the copyright policy; (d) the public training-data summary.

!!! tip "The open-source carve-out (Article 53(2)) — and its limits"

    If you release your model under a **free and open-source licence** that permits access, use, modification and distribution, **and** you publicly release the parameters (including weights), the architecture information, and the model-usage information, then obligations (a) and (b) — the technical documentation and the downstream-provider documentation — do not apply to you.

    Two limits matter enormously and are routinely missed. First, the carve-out **does not** cover (c) the copyright policy or (d) the public training-data summary: those apply to every GPAI provider, open-weight or not. Second, the carve-out **evaporates entirely** for a model with systemic risk. So an open-weight 100M-parameter model still owes the world a copyright policy and a training-data summary; an open-weight 500B-parameter frontier model owes everything. If you are open-weighting Stack-100M from Part XIV, this is precisely the obligation surface you inherit — see the note below.

### GPAI Systemic-Risk Obligations (Article 55)

For models above $10^{25}$ FLOPs, four *additional* obligations apply, in the order Article 55(1) lists them:

1. **Model evaluation, including adversarial testing** (Art. 55(1)(a)) — Evaluate the model per standardised protocols and state-of-the-art tools, including conducting and documenting adversarial testing (red-teaming), to identify and mitigate systemic risks.
2. **Systemic-risk assessment and mitigation** (Art. 55(1)(b)) — Assess and mitigate possible systemic risks at Union level, including those arising from development, market placement or use.
3. **Serious-incident reporting** (Art. 55(1)(c)) — Track, document, and report without undue delay to the AI Office (and, as appropriate, national competent authorities) information about serious incidents and corrective measures.
4. **Cybersecurity measures** (Art. 55(1)(d)) — Ensure an adequate level of cybersecurity protection for the model and its physical infrastructure.

A frequent citation error is worth flagging: **energy reporting is not in Article 55**. The obligation to document the "known or estimated energy consumption" of the model sits in **Annex XI**, the contents list for the Article 53(1)(a) technical documentation, so it lands on GPAI providers generally rather than only on systemic-risk providers. Article 55(1)(d) is cybersecurity, not energy.

Article 55(2) adds the compliance route that actually matters in practice: providers may rely on **codes of practice** to demonstrate compliance until a harmonised European standard exists. The Commission-endorsed **GPAI Code of Practice** (final version July 2025) is that route — it has a Transparency chapter (with a fill-in Model Documentation Form), a Copyright chapter, and a Safety and Security chapter that operationalises Article 55, including a Safety and Security Framework, a pre-deployment Model Report to the AI Office, and tiered initial-reporting deadlines for serious incidents (measured in days, tightest for incidents that are serious cybersecurity breaches of model controls). Signing the Code does not create a legal presumption of conformity the way a harmonised standard would, but it gives the AI Office a defined checklist and gives you legal certainty about what "adequate" means. Check the current text before relying on any specific deadline in it.

{{fig:gov-flop-systemic-risk-gate}}

!!! note "Where Stack-100M lands"

    Run the same arithmetic on the capstone model. Stack-100M is roughly $N = 1\times10^{8}$ parameters trained on $D = 2\times10^{10}$ tokens ([Data: Sourcing, Filtering, Dedup, Tokenize & Pack ~20B Tokens](../14-capstone/02-data-pipeline.html)):

    $$
    C = 6 \times 10^{8} \times 2\times10^{10} = 1.2\times10^{19}\ \text{FLOPs}
    $$

    That is about six orders of magnitude below the $10^{25}$ systemic-risk bar, and also below the roughly $10^{23}$-FLOP figure the Commission's 2025 GPAI guidelines propose as an *indicative* threshold for presuming a model is a general-purpose AI model at all. So Stack-100M is almost certainly not a GPAI model in the Act's sense, and Chapter V does not bite.

    That is a reason to practise the artefacts, not to skip them. Three things still apply in the real world. (1) If you publish the weights, downstream fine-tuners and deployers will ask for exactly the Article 53 evidence — model card, training-data summary, licence and rights basis — because *their* deployment may be high-risk even though your model is not. (2) The copyright duty in the DSM Directive applies to your crawl regardless of the Act: honour TDM opt-outs when you collect the 20 B tokens. (3) The compute, energy and data-provenance numbers are cheapest to capture during the run, which is why the capstone's cost accounting and reproducibility ledger ([Retrospective: Cost Accounting, Reproducibility, and the Path to 1B](../14-capstone/12-retrospective-and-scaleup.html)) doubles as a compliance record, and the honest-benchmark reporting in [Evaluation & Serving: Honest Benchmarks, int4 Quantization, and Running on a Laptop](../14-capstone/11-evaluation-and-serving.html) doubles as the eval-report evidence.

---

## High-Risk Application Requirements

When an LLM is deployed in a high-risk context (Annex III), the *deployer* (or developer, if they are also the deployer) must implement a compliance programme before placing the system on the EU market or putting it into service.

### Conformity Assessment

The deployer must perform a conformity assessment and register the system in the EU AI database before deployment. For most Annex III categories, this is a self-assessment. For biometric identification and law-enforcement use-cases, a notified body (third-party auditor) must be involved.

### Required Technical Controls


{{fig:gov-high-risk-requirements-grid}}


The **logging** requirement is operationally significant. Article 12 requires that high-risk AI systems be designed to automatically generate logs throughout their lifecycle — including: input data (or a reference to it), output data, the identity of persons or processes that invoked the system, and the date/time of operation.

Here is a minimal compliant logging schema for an LLM application, implemented as a structured JSON record:

```python
# eu_ai_act_logger.py
# Minimal logging record meeting EU AI Act Article 12 for a high-risk deployment.
# Writes one JSON object per request to a tamper-evident append-only log stream.

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AIActLogRecord:
    """
    One inference event, structured to meet Art. 12 (logging) and
    Art. 26(5) (transparency) of the EU AI Act.
    """
    # Unique identifier for this event (for incident correlation)
    event_id: str

    # UTC timestamp as Unix epoch with milliseconds
    timestamp_ms: float

    # Identity of the invoking system or user (pseudonymised where required)
    invoker_id: str

    # SHA-256 digest of the raw input text (avoids storing personal data
    # verbatim while still enabling reconstruction under legal obligation)
    input_sha256: str

    # Full output text — retained for audit (encrypt at rest)
    output_text: str

    # Model version string and inference parameters that determined the output
    model_version: str
    temperature: float
    max_tokens: int

    # The use-case context that triggered the deployment
    deployment_context: str

    # Optional: human reviewer decision (for human-oversight workflows)
    human_reviewer_id: Optional[str] = None
    human_decision: Optional[str] = None   # "approved", "rejected", "modified"
    human_decision_timestamp_ms: Optional[float] = None


def make_log_record(
    invoker_id: str,
    input_text: str,
    output_text: str,
    model_version: str,
    temperature: float,
    max_tokens: int,
    deployment_context: str,
) -> AIActLogRecord:
    """Construct a compliant log record from inference inputs/outputs."""
    input_bytes = input_text.encode("utf-8")
    input_digest = hashlib.sha256(input_bytes).hexdigest()

    return AIActLogRecord(
        event_id=str(uuid.uuid4()),
        timestamp_ms=time.time() * 1000,
        invoker_id=invoker_id,
        input_sha256=input_digest,
        output_text=output_text,
        model_version=model_version,
        temperature=temperature,
        max_tokens=max_tokens,
        deployment_context=deployment_context,
    )


def write_log_record(record: AIActLogRecord, log_path: str) -> None:
    """
    Append a JSON log record to an append-only file.
    In production, replace with a write-once object store (S3 Object Lock,
    Azure Immutable Blob, or an immutable audit log service).
    Include a chain hash for tamper evidence in high-assurance deployments.
    """
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    record = make_log_record(
        invoker_id="user:pseudonym-7f3a",   # hashed / pseudonymised PII
        input_text="Is this loan application likely to be approved?",
        output_text="Based on the supplied financial data, the risk score ...",
        model_version="credit-risk-llm-v2.1.3",
        temperature=0.0,
        max_tokens=512,
        deployment_context="eu-high-risk:credit-scoring:annex-iii-b5",
    )
    write_log_record(record, "/var/log/ai-act-audit.jsonl")
    print(f"Logged event {record.event_id}")
```

### Fines

From August 2026, enforcement is live. Fines are capped at:

- Up to **EUR 35 million or 7 % of global annual turnover** (whichever is higher) for violations of prohibited AI practices.
- Up to **EUR 15 million or 3 %** for other violations of the Act.
- Up to **EUR 7.5 million or 1.5 %** for providing incorrect information to authorities.

The EU AI Office has enforcement jurisdiction over GPAI models; national market surveillance authorities handle high-risk application violations.

---

## Model Cards and Datasheets: The Documentation Artefacts

Two documentation artefacts have become the standard technical interface for governance: **model cards** (Mitchell et al., 2019) and **datasheets for datasets** (Gebru et al., 2021). Both pre-date regulation but now serve as the primary way to satisfy Article 53 technical documentation requirements.

### Model Card Structure

A production-quality model card for a GPAI model covers:

```yaml
# model-card.yaml
# Structured model card following Hugging Face / EU AI Act Article 53 conventions.

model_name: "ExampleLLM-70B"
model_version: "1.2.0"
release_date: "2025-09-01"

# ── Identity ─────────────────────────────────────────────────────────────────
provider:
  name: "ExampleCorp"
  contact: "ai-governance@example.com"
  eu_establishment: "ExampleCorp GmbH, Berlin, DE"

# ── Architecture ─────────────────────────────────────────────────────────────
architecture:
  family: "Decoder-only transformer"
  parameter_count: "70 billion"
  context_length: 131072
  tokenizer: "BPE, 128 k vocab"
  precision: "bfloat16"

# ── Training ─────────────────────────────────────────────────────────────────
training:
  compute_flops: "~6e23"           # below 1e25 systemic-risk threshold
  hardware: "4096 x H100 SXM5"
  duration_days: 42
  training_objective: "Next-token prediction (causal LM)"
  post_training: ["SFT", "RLHF/DPO"]

# ── Training Data (Article 53(1)(d) summary) ─────────────────────────────────
training_data:
  summary_url: "https://example.com/model-docs/training-data-summary-v1.2.0.pdf"
  languages: ["en", "de", "fr", "es", "zh", "ja", "ko", "ar"]  # top 8 of 50+
  total_tokens: "2 trillion (estimated)"
  sources:
    - name: "Common Crawl (filtered)"
      license: "Public domain / robots.txt compliant"
      fraction: "~45 %"
    - name: "Curated books corpus"
      license: "Licensed; see data-rights-register.csv"
      fraction: "~10 %"
    - name: "Code repositories"
      license: "OSI-approved licenses only"
      fraction: "~15 %"
    - name: "Wikipedia / Wikidata"
      license: "CC-BY-SA 4.0"
      fraction: "~5 %"
    - name: "Synthetic instruction data"
      license: "Proprietary"
      fraction: "~25 %"
  personal_data_handling: >
    Data was filtered to remove documents containing email addresses,
    phone numbers, and national ID patterns. Differential-privacy noise
    was NOT applied at pretraining; see privacy impact assessment v1.2.
  opt_out_mechanism: "https://example.com/data-opt-out"

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluations:
  - benchmark: "MMLU"
    score: "see eval-report-v1.2.0.pdf"   # we do not fabricate numbers
    methodology: "5-shot"
  - benchmark: "HumanEval"
    score: "see eval-report-v1.2.0.pdf"
    methodology: "pass@1, greedy"
  - benchmark: "MT-Bench"
    score: "see eval-report-v1.2.0.pdf"

# ── Safety evaluations (expected of GPAI providers) ──────────────────────────
safety_evals:
  - name: "Dangerous capabilities (bio, chem, cyber, radiological)"
    methodology: "Internal red-team + third-party assessment"
    passed: true
  - name: "Bias and fairness (BBQ, WinoBias)"
    methodology: "Automated + human review"
    result: "see fairness-report-v1.2.0.pdf"
  - name: "Adversarial robustness"
    methodology: "AutoAttack, PAIR jailbreak suite"
    result: "see adversarial-report-v1.2.0.pdf"

# ── Intended Use ─────────────────────────────────────────────────────────────
intended_use:
  primary: "General-purpose text generation via API"
  out_of_scope:
    - "Autonomous medical diagnosis without human oversight"
    - "Real-time biometric identification"
    - "Law enforcement decision-making without human review"

# ── Known Limitations ────────────────────────────────────────────────────────
limitations:
  - "Knowledge cut-off: 2025-07-01; no awareness of later events"
  - "Hallucination rate on low-resource languages estimated higher than on English"
  - "May reproduce biases present in training data"

# ── EU AI Act Compliance Status ──────────────────────────────────────────────
eu_ai_act:
  gpai_model: true
  systemic_risk: false          # compute < 1e25 FLOPs
  copyright_policy_url: "https://example.com/model-docs/copyright-policy.pdf"
  technical_documentation_url: "https://example.com/model-docs/tech-doc-v1.2.0.pdf"
  ai_office_registration_id: "EUAIO-GPAI-2025-00042"   # fictional example
```

#### Generating the card with `huggingface_hub`

You do not have to invent the serialisation format. The Hugging Face Hub's model card is a Markdown file (`README.md`) with a YAML front-matter block, and `huggingface_hub` gives you a typed API for it — which means the card can be emitted by the same job that finishes training, instead of being written by hand weeks later:

```python
# generate_model_card.py
# Emit a Hub-compatible model card whose YAML front matter carries the
# structured governance metadata, then attach the AI-Act sections as body text.
#   pip install "huggingface_hub>=0.24"

from huggingface_hub import ModelCard, ModelCardData, EvalResult

card_data = ModelCardData(
    model_name="Stack-100M",
    license="apache-2.0",                 # SPDX id; the weights licence
    language=["en"],
    library_name="transformers",
    tags=["eu-ai-act", "governance", "open-weights"],
    datasets=["HuggingFaceFW/fineweb-edu"],   # declared training sources
    eval_results=[
        EvalResult(
            task_type="text-generation",
            dataset_type="hellaswag",
            dataset_name="HellaSwag",
            metric_type="accuracy",
            metric_value=0.0,             # fill from your eval harness output
        )
    ],
)

# `from_template` with no template_path uses the Hub's default card template.
card = ModelCard.from_template(
    card_data,
    model_id="Stack-100M",
    developers="ExampleCorp",
    model_description="A 100M-parameter decoder-only LM trained from scratch.",
)

# Append the governance sections the default template does not cover.
card.text += (
    "\n\n## EU AI Act status\n"
    "- Training compute (6ND estimate): 1.2e19 FLOPs — below the 1e25 "
    "systemic-risk threshold and below the indicative GPAI threshold.\n"
    "- Training-data summary: see `training-data-summary.md`.\n"
    "- Copyright policy: TDM opt-outs honoured at crawl time; see "
    "`rights-register.csv`.\n"
)

card.save("README.md")          # or: card.push_to_hub("ExampleCorp/Stack-100M")
```

The same package exposes `DatasetCard` / `DatasetCardData` for the dataset side. Two other tools are worth wiring in: the MLCommons **Croissant** metadata format (`mlcroissant`), which the Hub emits for datasets and which gives you a machine-readable description of fields, licences and provenance; and **`sigstore/model-transparency`** (the OpenSSF model-signing project), which signs model artefacts so a downstream user can verify the weights they loaded are the ones your card describes — the supply-chain half of "chain of custody".

### Dataset Datasheets

Gebru et al.'s datasheet framework asks: *Motivation, Composition, Collection Process, Preprocessing, Uses, Distribution, Maintenance*. The EU AI Act Article 53 training-data summary maps almost perfectly onto this structure. The key addition that regulation requires is an explicit rights record: for each data source, who holds the rights, what licence applies, and whether an Article 4(3) reservation (opt-out from text-and-data mining) was filed against that source.

```python
# rights_register.py
# Minimal data-rights register for EU AI Act Art. 53(1)(c) copyright compliance.
# Production version should be stored in a version-controlled database,
# not a plain Python dict.

import csv
import io
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataSourceRecord:
    """
    One entry in the training-data rights register.
    Mirrors the structure recommended by the European AI Office
    code-of-practice drafts (2025).
    """
    source_id: str              # e.g. "CC-2023-11-EN"
    source_name: str
    url: str
    crawl_date: str             # ISO 8601

    # Copyright / licencing
    licence: str                # SPDX identifier or "Proprietary" or "Unknown"
    rightsholder: str
    text_data_mining_exception_applies: bool   # EU DSM Directive Art. 4
    opt_out_detected: bool      # Did the rightsholder file an Art. 4(3) opt-out?
    opt_out_respected: bool     # Did we exclude the source upon detection?

    # Data quality
    language_codes: List[str] = field(default_factory=list)
    approximate_tokens: int = 0
    personal_data_present: bool = False
    deduplication_method: str = "MinHash LSH"

    def is_compliant(self) -> bool:
        """
        A source is compliant if either:
          (a) The text-and-data mining exception applies and no opt-out exists, or
          (b) We have an explicit licence permitting ML training, or
          (c) An opt-out existed and we excluded the source.
        """
        if self.opt_out_detected and not self.opt_out_respected:
            return False   # Violation: ignored an opt-out
        if self.text_data_mining_exception_applies and not self.opt_out_detected:
            return True    # Clean TDM exception
        # Otherwise fall through to licence check
        compliant_licences = {
            "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0",
            "MIT", "Apache-2.0", "GPL-2.0-only", "GPL-3.0-only",
            "Public Domain",
        }
        return self.licence in compliant_licences


def export_rights_register(records: List[DataSourceRecord]) -> str:
    """Export the register as CSV for submission to the European AI Office."""
    buf = io.StringIO()
    fields = [
        "source_id", "source_name", "url", "crawl_date", "licence",
        "rightsholder", "text_data_mining_exception_applies",
        "opt_out_detected", "opt_out_respected", "compliant",
        "approximate_tokens",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for rec in records:
        writer.writerow({
            "source_id": rec.source_id,
            "source_name": rec.source_name,
            "url": rec.url,
            "crawl_date": rec.crawl_date,
            "licence": rec.licence,
            "rightsholder": rec.rightsholder,
            "text_data_mining_exception_applies": rec.text_data_mining_exception_applies,
            "opt_out_detected": rec.opt_out_detected,
            "opt_out_respected": rec.opt_out_respected,
            "compliant": rec.is_compliant(),
            "approximate_tokens": rec.approximate_tokens,
        })
    return buf.getvalue()
```

---

## Serious-Incident Reporting

**Article 73** of the EU AI Act (numbered Article 62 in earlier drafts — a stale citation you will still see in blog posts and in vendor compliance decks) requires providers of high-risk AI systems to notify the market surveillance authority of the Member State where the incident occurred of *serious incidents*. Article 3(49) defines a serious incident as one that directly or indirectly leads to: the death of a person or serious harm to health; a serious and irreversible disruption of the management or operation of critical infrastructure; infringement of Union law obligations protecting fundamental rights; or serious harm to property or the environment.

The reporting clock is tiered rather than flat. The outer limit is **15 days** after the provider becomes aware; it tightens to **10 days** where the incident involves a person's death, and to **2 days** for a widespread infringement or a serious and irreversible disruption of critical infrastructure. Where the full picture is not yet available, the Act contemplates an **initial incomplete report followed by a complete one** — build your pipeline to send a partial notification on the deadline rather than to wait for a finished root-cause analysis.

For GPAI providers with systemic risk, **Article 55(1)(c)** separately requires tracking, documenting and reporting serious incidents and corrective measures to the European AI Office "without undue delay". The Act itself puts no number on it; the GPAI Code of Practice's Safety and Security chapter is where the concrete day counts live, and the tightest of them is on the order of a couple of days for incidents that constitute serious cybersecurity breaches of model controls. The code below uses **2 days** as the systemic-risk budget and 15 days as the high-risk budget; treat those as configuration, keyed to whichever instrument currently binds you, not as constants of nature.

### Building a Compliant Incident Pipeline

```python
# incident_reporter.py
# Production-grade skeleton for EU AI Act serious-incident reporting.
# Integrates with an existing observability stack (Prometheus / PagerDuty).

import enum
import json
import smtplib
import time
import uuid
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from typing import Optional


class IncidentSeverity(enum.Enum):
    """
    Severity classification mapping to regulatory reporting thresholds.
    """
    MINOR = "minor"              # Internal only; no external reporting required
    SIGNIFICANT = "significant"  # Log; 72-hour internal review required
    SERIOUS = "serious"          # Art. 73 notification to national authority (≤15 days;
                                 # ≤10 if a death, ≤2 if critical-infrastructure disruption)
    SYSTEMIC = "systemic"        # Art. 55(1)(c) notification to EU AI Office (Code of
                                 # Practice deadline; ~2 days for control breaches)


@dataclass
class AIIncidentReport:
    incident_id: str
    detection_timestamp_ms: float
    severity: IncidentSeverity

    # Description fields for regulatory notification
    description: str
    affected_system: str           # model_version + deployment_context
    number_of_affected_persons: Optional[int]
    harm_category: str             # "health", "property", "fundamental_rights", etc.
    corrective_measures_taken: str
    ongoing: bool

    # Internal tracking
    detected_by: str               # "automated_monitor", "user_report", "red_team"
    assigned_to: str
    notified_authority: Optional[str] = None
    notification_timestamp_ms: Optional[float] = None


def triage_incident(
    description: str,
    harm_indicators: dict,
    system_id: str,
) -> AIIncidentReport:
    """
    Triage an incoming event and assign severity.
    harm_indicators keys: death, serious_injury, service_disruption,
    fundamental_rights_violation, property_damage (all bool).
    """
    is_serious = any([
        harm_indicators.get("death"),
        harm_indicators.get("serious_injury"),
        harm_indicators.get("fundamental_rights_violation"),
    ])
    is_systemic = harm_indicators.get("broad_societal_impact")

    if is_systemic:
        severity = IncidentSeverity.SYSTEMIC
    elif is_serious:
        severity = IncidentSeverity.SERIOUS
    elif harm_indicators.get("service_disruption") or harm_indicators.get("property_damage"):
        severity = IncidentSeverity.SIGNIFICANT
    else:
        severity = IncidentSeverity.MINOR

    return AIIncidentReport(
        incident_id=str(uuid.uuid4()),
        detection_timestamp_ms=time.time() * 1000,
        severity=severity,
        description=description,
        affected_system=system_id,
        number_of_affected_persons=harm_indicators.get("affected_count"),
        harm_category=_infer_harm_category(harm_indicators),
        corrective_measures_taken="Under investigation",
        ongoing=True,
        detected_by="automated_monitor",
        assigned_to="ai-safety-team@example.com",
    )


def _infer_harm_category(harm_indicators: dict) -> str:
    if harm_indicators.get("death") or harm_indicators.get("serious_injury"):
        return "health"
    if harm_indicators.get("fundamental_rights_violation"):
        return "fundamental_rights"
    if harm_indicators.get("service_disruption"):
        return "essential_services"
    return "property"


def notify_authority(
    report: AIIncidentReport,
    smtp_host: str,
    authority_email: str,
) -> None:
    """
    Send structured incident notification email to the relevant authority.
    Replace with the AI Office / national-authority reporting portal API
    when one is published.
    Deadline: SYSTEMIC ~2 days (Code of Practice); SERIOUS <=15 days
    (Art. 73; tighter for deaths and critical-infrastructure disruption).
    """
    body = json.dumps(asdict(report), indent=2, default=str)
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = (
        f"[AI Act Incident Notification] {report.severity.value.upper()} "
        f"– {report.incident_id}"
    )
    msg["From"] = "ai-governance@example.com"
    msg["To"] = authority_email

    with smtplib.SMTP(smtp_host) as s:
        s.sendmail(msg["From"], [msg["To"]], msg.as_string())

    report.notified_authority = authority_email
    report.notification_timestamp_ms = time.time() * 1000
```

!!! warning "Common pitfall"

    Many teams conflate their general security-incident response process with AI-Act incident reporting. The key difference: AI-Act incidents are triggered by *harm or potential harm to people*, not by service outages or security breaches per se. A DDOS attack on your inference API is a security incident; a model that caused a user to self-harm following biased mental-health advice is an AI Act serious incident. Build separate triage paths — but wire them to the same event bus, because the two do intersect (an outage of an AI system embedded in critical infrastructure *is* an Article 73 serious incident, and a breach of model-weight controls at a systemic-risk provider *is* an Article 55 one). The rule is separate triage criteria and separate deadlines, not separate telemetry. See [Reliability Engineering for LLM Systems: SLOs & Incident Response](../12-production-mlops/08-reliability-engineering.html) for the SRE side of the same pipeline.

{{fig:gov-incident-severity-ladder}}

---

## Transparency and Eval Reporting

Beyond the model card, two categories of transparency reporting are increasingly required or expected: **pre-deployment eval reports** and **ongoing transparency reports**.

### Pre-Deployment Eval Reports

Before any deployment in a high-risk context, the eval suite must be documented with sufficient detail to allow reproducibility. This connects to the work in [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html) and [Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html). A compliant eval report structure:

```json
{
  "report_type": "pre_deployment_eval",
  "model": "ExampleLLM-70B-v1.2.0",
  "deployment_context": "eu-high-risk:employment-screening:annex-iii-b4",
  "eval_date": "2025-08-15",
  "evaluator": {
    "team": "AI Safety & Compliance",
    "independence": "internal"
  },
  "benchmarks": [
    {
      "name": "Employment-decision bias (race/gender)",
      "method": "Counterfactual data augmentation; 1000 test pairs",
      "metric": "Demographic parity difference",
      "result_summary": "< 5% disparity on held-out evaluation set",
      "pass_threshold": "< 10%",
      "status": "PASS"
    },
    {
      "name": "Adversarial prompt robustness",
      "method": "PAIR jailbreak suite + human red-team (10 person-hours)",
      "result_summary": "No systematic safety failures identified",
      "status": "PASS"
    },
    {
      "name": "Factual accuracy on domain-specific questions",
      "method": "Expert-labelled Q&A set (n=500)",
      "result_summary": "See supplementary table A",
      "status": "PASS"
    }
  ],
  "human_oversight_mechanism": "All model outputs reviewed by HR specialist before actioning",
  "limitations_acknowledged": [
    "Performance on non-EU legal frameworks not evaluated",
    "Intersectional bias (race × gender) not fully characterised"
  ],
  "sign_off": {
    "ai_officer": "Jane Smith",
    "date": "2025-08-20"
  }
}
```

### Ongoing Transparency Reports

The Act does not mandate a general annual "transparency report" for GPAI providers — the only artefact Article 53 requires you to *publish* is the training-data summary (Art. 53(1)(d)); the technical documentation goes to the AI Office on request, not to the public. But the combination of the GPAI Code of Practice (which expects a maintained Model Documentation Form and, for systemic-risk models, a Safety and Security Model Report to the AI Office), California SB 53's published frontier framework and incident reporting, and the voluntary lab frameworks (Frontier Safety Framework, Responsible Scaling Policy, Preparedness Framework) has converged on a recurring public report as the de facto norm. Key sections in such a report:

1. **Model population summary** — All live models, versions, compute tier, and GPAI/systemic-risk classification.
2. **Incident log summary** — Aggregated statistics on serious-incident notifications (without PII).
3. **Red-team summary** — High-level results of adversarial evaluations since last report.
4. **Copyright and data-rights updates** — Changes to training-data composition, new opt-out compliance actions.
5. **Energy and compute disclosure** — Training compute and known or estimated energy consumption, per Annex XI (the contents list for the Article 53(1)(a) technical documentation).

---

## NIST AI RMF and ISO/IEC 42001

The EU AI Act specifies *what* to do. Two complementary frameworks specify *how* to build an organisation that does it reliably.

### NIST AI Risk Management Framework (AI RMF 1.0, 2023)

The NIST AI RMF is structured around four core functions:

| Function | Description | Key activities |
|---|---|---|
| **GOVERN** | Establish policies, roles, culture | Appoint an AI risk owner; define AI ethics principles; set risk tolerance |
| **MAP** | Contextualise risk | Identify AI use-cases; map to risk categories; identify affected stakeholders |
| **MEASURE** | Analyse and assess risk | Run evals; test for bias; monitor in deployment; document results |
| **MANAGE** | Prioritise and treat risk | Risk treatment plans; human oversight; incident response; decommission plans |

The AI RMF has no enforcement teeth in the US (it is voluntary), but it is referenced in US government procurement requirements and increasingly incorporated into enterprise supplier contracts. It is also the operational backbone many companies use to satisfy EU AI Act obligations — the NIST RMF's GOVERN and MANAGE functions map directly onto the Act's risk-management system requirements.

The complementary *NIST AI RMF Generative AI Profile* (NIST AI 600-1, published 2024) adapts the framework to foundation models and covers: hallucination, data privacy, harmful content, intellectual property concerns, information integrity, and homogenisation risk.

### ISO/IEC 42001:2023 — AI Management System

ISO/IEC 42001 is the AI equivalent of ISO 27001 (information security). It specifies requirements for an **AI Management System (AIMS)** that an organisation can implement and get independently certified against.

Its structure follows the standard ISO high-level structure (Annex SL):


{{fig:gov-iso-42001-clause-stack}}


For an LLM developer, the critical operational clauses are **Clause 8** (which requires documented AI system lifecycle controls — design, training data governance, testing, deployment, monitoring, decommissioning) and **Clause 9** (which requires a formal internal audit of those controls at planned intervals).

ISO/IEC 42001 certification signals to enterprise customers, regulators, and insurers that your AI governance is auditable, repeatable, and systematically managed — not ad hoc. The European Commission's standardisation mandate (Article 40) invites CEN/CENELEC to produce harmonised standards, and ISO/IEC 42001 is a strong candidate to become a presumption-of-conformity standard for the EU AI Act's risk-management requirements.

!!! note "Relationship between the three frameworks"

    Think of the three frameworks as nested: **ISO/IEC 42001** gives you the governance scaffolding (processes, roles, documentation); the **NIST AI RMF** provides the risk vocabulary and the MAP-MEASURE-MANAGE methodology; the **EU AI Act** specifies the legal floor — the minimum obligations you must satisfy within that scaffolding. Being ISO 42001-certified and NIST-aligned does not automatically mean EU-compliant, but it makes compliance dramatically easier to demonstrate.

---

## Engineering the Compliance Stack

Governance is not just policy — it is implemented in code, infrastructure, and process. Here is how the engineering artefacts connect:


{{fig:gov-compliance-engineering-stack}}


The key engineering insight is that most compliance artefacts should be **generated automatically** from metadata already produced during training and evaluation. If you wait until post-hoc to reconstruct training data summaries or compute counts, the records will be incomplete. Instrument your training pipeline to emit structured compliance metadata from day one.

Here is a minimal FLOP counter that emits a systemic-risk flag at training time:

```python
# flop_tracker.py
# Accumulate FLOPs across training steps and emit an alert if the
# EU AI Act systemic-risk threshold (1e25 FLOPs) is approached or crossed.

import logging

SYSTEMIC_RISK_THRESHOLD = 1e25   # EU AI Act Art. 51(2) presumption threshold
WARNING_FRACTION = 0.8           # Warn at 80% of threshold


class FlopTracker:
    """
    Tracks accumulated floating-point operations during a training run.

    Usage:
        tracker = FlopTracker(model_params=70e9)
        for batch in dataloader:
            flops_this_step = tracker.step(tokens_in_batch=batch.numel())
            if tracker.systemic_risk_reached():
                trigger_compliance_workflow()
    """

    def __init__(self, model_params: float):
        """
        model_params: number of trainable parameters (float, e.g. 70e9)
        """
        self.model_params = model_params
        self.total_flops: float = 0.0
        self._warned = False
        self._logger = logging.getLogger("flop_tracker")

    def step(self, tokens_in_batch: int) -> float:
        """
        Add FLOPs for one forward+backward pass over a batch.
        Approximation: 6 * N * T (Kaplan et al. / Chinchilla convention).
        Returns the incremental FLOPs for this step.
        """
        step_flops = 6.0 * self.model_params * tokens_in_batch
        self.total_flops += step_flops

        # Warn at 80% of threshold
        if (not self._warned
                and self.total_flops >= SYSTEMIC_RISK_THRESHOLD * WARNING_FRACTION):
            self._logger.warning(
                "FlopTracker: Training compute at %.2e FLOPs — approaching "
                "EU AI Act systemic-risk threshold (%.0e). "
                "Initiate systemic-risk compliance workflow.",
                self.total_flops, SYSTEMIC_RISK_THRESHOLD,
            )
            self._warned = True

        return step_flops

    def systemic_risk_reached(self) -> bool:
        """Returns True if training compute has met or exceeded the threshold."""
        return self.total_flops >= SYSTEMIC_RISK_THRESHOLD

    def summary(self) -> dict:
        """Return a serialisable summary for model card generation."""
        return {
            "total_flops": self.total_flops,
            "systemic_risk_threshold": SYSTEMIC_RISK_THRESHOLD,
            "systemic_risk_flag": self.systemic_risk_reached(),
            "fraction_of_threshold": self.total_flops / SYSTEMIC_RISK_THRESHOLD,
        }
```

### Measuring, Not Estimating: `FlopCounterMode` and `codecarbon`

The $6ND$ rule is an *estimate*, and a regulator asking "how did you arrive at that number?" deserves a better answer than "a scaling-laws paper". Two open-source tools turn the two headline compliance numbers — cumulative training FLOPs and training energy — into measurements you can defend.

PyTorch ships a FLOP counter as a dispatch-mode context manager. It counts the actual matmul/convolution FLOPs executed under it (including the backward pass, if the backward runs inside the block), which is exactly the "cumulative amount of computation used for training" the Act asks about, and it will show you how far off $6ND$ is for your architecture — MoE routing, attention at long context, and tied embeddings all move the ratio.

```python
# measured_flops.py
# Calibrate the analytic 6ND estimate against measured FLOPs for one step.
#   requires torch >= 2.1
import torch
from torch.utils.flop_counter import FlopCounterMode

model = ...            # your nn.Module
batch = ...            # one tokenised batch, shape (B, T)

with FlopCounterMode(display=False) as fcm:
    loss = model(batch).logits.float().mean()
    loss.backward()                      # counted too: fwd + bwd in one block

measured = fcm.get_total_flops()         # FLOPs for this fwd+bwd step
n_tokens = batch.numel()
analytic = 6.0 * sum(p.numel() for p in model.parameters()) * n_tokens
print(f"measured={measured:.3e}  analytic 6ND={analytic:.3e}  "
      f"ratio={measured / analytic:.3f}")
# Record the ratio once, then use `analytic * ratio` for the whole run so you
# are not paying the counter's overhead on every step.
```

For energy, **`codecarbon`** is the standard open-source instrument: it samples NVIDIA GPU power via NVML, CPU/RAM power, and applies a regional grid carbon intensity, writing a CSV you can attach verbatim to the Annex XI energy field.

```python
# track_energy.py
#   pip install codecarbon
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="stack-100m-pretrain",
                           output_dir="./compliance", log_level="error")
tracker.start()
try:
    train()                    # your training loop
finally:
    emissions_kg = tracker.stop()   # kg CO2eq; energy in kWh lands in the CSV
print(f"run emissions: {emissions_kg:.3f} kg CO2eq")
```

Alternatives with the same role: **`zeus`** (energy measurement and energy-optimal DVFS for DL training) and, if you already run a Prometheus stack, NVIDIA's **DCGM exporter** scraping `DCGM_FI_DEV_POWER_USAGE` and integrating over the run — see [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html). Whichever you pick, log the number *per run*, tagged with the run ID that also tags your checkpoints; reconstructing energy after the cluster has been reallocated is impossible.

---

## Copyright, Provenance, and the Chain of Custody

Chapter [Watermarking, Provenance & AI-Content Detection](../13-interp-safety-gov/04-watermarking-provenance.html) covers the technical watermarking side. Here we focus on the *legal* provenance chain.

### The EU Copyright Dimension

The EU Copyright in the Digital Single Market Directive (2019/790), Articles 3 and 4, create a text-and-data mining (TDM) exception that permits ML training on lawfully accessed content — unless the rightsholder has "expressly reserved" their rights (an opt-out, typically filed via `robots.txt` or a machine-readable TDM reservation attached to the work).

For LLM developers, this creates a duty to:

1. **Check opt-out signals** at crawl time. Three mechanisms are in real use: (a) `robots.txt` disallow rules aimed at named AI crawlers (`GPTBot`, `ClaudeBot`, `Google-Extended`, `CCBot`, `Bytespider`, …); (b) the **W3C TDM Reservation Protocol (TDMRep)**, whose signal `tdm-reservation: 1` ("rights reserved") can be carried in an HTTP header, an HTML `<meta name="tdm-reservation">` tag, or a site-wide `/.well-known/tdmrep.json` document, optionally paired with `tdm-policy` pointing at licensing terms; and (c) licence metadata attached to the work itself. A fourth is stabilising: the IETF **AI Preferences (`aipref`)** working group is standardising a vocabulary for expressing training-usage preferences, so expect the signal surface to consolidate rather than shrink.
2. **Exclude opted-out sources** from training data — and note that "lawfully accessible" is a precondition of the exception in the first place, so paywalled or pirated corpora do not become lawful just because no opt-out was filed.
3. **Maintain a rights register** documenting the basis for including each source.
4. **Retain evidence** of opt-out checks, preferably a hashed and timestamped record of the `robots.txt` / `tdmrep.json` as it existed at crawl time. Opt-outs are not retroactive in practice — you must be able to show what the signal said *on the day you crawled*, which is why the check result belongs in the per-document provenance record, not in a separate audit spreadsheet.

The following snippet shows how to check TDM reservation signals when building a crawler:

```python
# tdm_checker.py
# Check TDM reservation signals before including a URL's content in training data.
# Should be called as part of the data collection pipeline.

import urllib.robotparser


# W3C TDM Reservation Protocol (TDMRep). The signal is a VALUE, not a
# presence check: "1" means rights are reserved (opt-out), "0" means not
# reserved. The same key appears as an HTTP header, an HTML <meta> name,
# and a field in /.well-known/tdmrep.json.
TDM_RESERVATION_HEADERS = ("tdm-reservation", "x-tdm-reservation")


def check_tdm_opt_out(url: str, headers: dict, robots_txt: str) -> dict:
    """
    Returns a compliance record indicating whether the rightsholder has
    filed an opt-out and whether we should exclude the URL.

    url        — The URL being evaluated.
    headers    — HTTP response headers (lowercase keys).
    robots_txt — The raw robots.txt text fetched from the site's root.
    """
    result = {
        "url": url,
        "tdm_reservation_header_present": False,
        "robots_txt_disallows_tdm": False,
        "should_exclude": False,
        "reason": None,
    }

    # 1. Check TDMRep reservation headers. Reserved iff the value is "1".
    for h in TDM_RESERVATION_HEADERS:
        val = headers.get(h)
        if val is not None:
            result["tdm_reservation_header_present"] = True
            if val.strip() == "1":
                result["should_exclude"] = True
                result["reason"] = f"TDMRep opt-out: {h}={val}"
                return result

    # 2. Check robots.txt for AI training crawlers.
    # Conservative policy: if ANY of the user-agent tokens we crawl under is
    # disallowed for this path, treat the source as opted out. Note that
    # robots.txt is a *site* signal while TDMRep can be per-resource, so a
    # production crawler evaluates both and takes the union of exclusions.
    ai_crawlers = ["GPTBot", "CCBot", "ClaudeBot", "Google-Extended", "Bytespider"]
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(robots_txt.splitlines())

    for crawler in ai_crawlers:
        if not rp.can_fetch(crawler, url):
            result["robots_txt_disallows_tdm"] = True
            result["should_exclude"] = True
            result["reason"] = f"robots.txt disallows {crawler}"
            return result

    # 3. Remaining TDMRep surfaces the caller should also feed in:
    #    - HTML <meta name="tdm-reservation" content="1"> in the page head
    #    - the site-wide /.well-known/tdmrep.json document, whose entries
    #      match URL path prefixes and may carry a tdm-policy URL
    #    Both are omitted here for brevity; see the W3C TDMRep specification.

    result["should_exclude"] = False
    result["reason"] = "No opt-out signal detected"
    return result
```

### Provenance Metadata in the Pretraining Pipeline

Every document that enters training should carry a **provenance record** — source URL, crawl timestamp, licence, opt-out check result — stored alongside the tokenised data. This is cheap at training time and invaluable when a regulator or rightsholder later asks "did this document go into your training data?"

Concretely, this is a field you add to the document schema in your extraction/filtering pipeline. In **`datatrove`** (the HF pipeline used to build FineWeb) every `Document` carries a free-form `metadata` dict that survives each `PipelineStep`, so a custom filter can stamp `metadata["tdm"] = check_tdm_opt_out(...)` at the WARC-reading stage and every downstream stage — dedup, quality filtering, tokenisation — will carry it through to the final shard index. NVIDIA **NeMo-Curator** and **Dolma** expose the same idea under different names. The rule of thumb: provenance must be attached at the *earliest* stage and must never be dropped by a stage that rewrites text, because after tokenisation and packing the association between a token span and its source document is effectively unrecoverable.

See [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) for the broader data pipeline engineering, [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) for deduplication methods that also help with copyright compliance, and [Data: Sourcing, Filtering, Dedup, Tokenize & Pack ~20B Tokens](../14-capstone/02-data-pipeline.html) for the capstone pipeline where you would actually add this field.

---

!!! interview "Interview Corner"

    **Q:** A regulator asks your team to demonstrate that your 400 B parameter LLM complies with the EU AI Act. Walk through the obligations that apply and the artefacts you would produce.

    **A:** First, calculate training compute: with roughly $C = 6 \times 4 \times 10^{11} \times 1.5 \times 10^{13} \approx 3.6 \times 10^{25}$ FLOPs, the model crosses the $10^{25}$ systemic-risk threshold, so both base GPAI obligations (Art. 53) and systemic-risk obligations (Art. 55) apply.

    Base GPAI artefacts: (1) Technical documentation covering architecture, training data, compute, evaluations, and limitations; (2) a training-data summary published publicly, covering data sources, licence basis, and opt-out compliance; (3) a copyright policy; (4) downstream-deployer documentation.

    Systemic-risk additions (Art. 55(1), letters (a)–(d)): (1) model evaluation including documented adversarial testing; (2) a systemic-risk assessment and mitigation record; (3) a serious-incident pipeline reporting to the EU AI Office without undue delay, sized to the day-count deadlines in the GPAI Code of Practice; (4) cybersecurity protection for the weights and the infrastructure. I would note that energy consumption is an Annex XI documentation item that applies to us as a GPAI provider generally, not an Article 55 item — getting that right signals I have read the instrument rather than a summary of it. Because we are not open-weight, the Article 53(2) carve-out does not help us; even if we were, it would not survive the systemic-risk classification.

    I would present: the model card YAML, the rights register CSV, the pre-deployment eval report JSON, the audit log schema, the incident-triage code, and the FlopTracker summary. I would also produce the EU AI Office registration record and confirm Art. 53 training-data summary has been published.

---

!!! key "Key Takeaways"

    - The EU AI Act phases in obligations over 2025–2027; the most operationally significant dates are **Aug 2025** (GPAI model duties) and **Aug 2026** (high-risk application enforcement with fines up to 7% of global turnover).
    - The **systemic-risk threshold** is $10^{25}$ FLOPs of training compute (Art. 51(2)). Models below it still carry the Art. 53 documentation, copyright and training-data-summary obligations — including the Annex XI energy field; models above it add Art. 55's model evaluation and adversarial testing, systemic-risk mitigation, incident reporting to the EU AI Office, and cybersecurity.
    - The **Article 53(2) open-source carve-out** drops the technical-documentation and downstream-provider duties for genuinely open-weight releases, but never drops the copyright policy or the public training-data summary, and disappears entirely once a model crosses the systemic-risk threshold.
    - **Article 50** turns the "limited-risk" tier into real work from Aug 2026: disclose the AI, and mark synthetic output in a machine-readable way (C2PA content credentials plus text watermarking are the two practical layers).
    - **Compliance artefacts** — model cards, training-data summaries, rights registers, eval reports, and audit logs — should be generated automatically from metadata captured during training and evaluation, not reconstructed after the fact.
    - **Audit logs** for high-risk deployments must include input references, output text, invoker identity, and timestamps; use append-only or immutable storage.
    - **Serious-incident reporting** is triggered by harm to people, not system outages; build a separate triage pipeline distinct from general SRE incident response.
    - The **NIST AI RMF** (GOVERN/MAP/MEASURE/MANAGE) and **ISO/IEC 42001** are complementary: the RMF gives the risk vocabulary and workflow; ISO 42001 gives the auditable management-system scaffold; the EU AI Act sets the legal floor.
    - **Copyright compliance** requires checking TDM opt-out signals at crawl time (`robots.txt`, TDM reservation headers), maintaining a rights register, and retaining timestamped evidence of those checks.
    - **Model cards** (Mitchell et al., 2019) and **datasheets for datasets** (Gebru et al., 2021) are the primary documentation artefacts; the EU AI Act training-data summary maps directly onto the datasheet structure.

---

!!! sota "State of the Art & Resources (2026)"
    AI governance and compliance has rapidly moved from voluntary guidance to binding law: the EU AI Act's GPAI-model obligations have been in force since Aug 2025 (with the Commission-endorsed GPAI Code of Practice as the primary compliance route), and the high-risk application and Article 50 transparency regimes — with fines up to 7% of global turnover — were legislated to take effect Aug 2026, though the Commission's late-2025 "Digital Omnibus" simplification proposal sought to postpone parts of that regime, so verify the current consolidated timetable before you plan against it. Meanwhile the US layer has shifted to the states (California SB 53's frontier-developer transparency and incident reporting; Colorado's repeatedly delayed AI Act), and the NIST AI RMF and ISO/IEC 42001 have become the operational backbone that organisations use to satisfy all of these at once. The resources below cover the foundational papers, the primary regulatory texts, and the open-source tooling engineers need to build compliant systems.

    **Foundational work**

    - [Mitchell et al., *Model Cards for Model Reporting* (2019)](https://arxiv.org/abs/1810.03993) — the seminal paper that defined the model card format now codified in EU AI Act Article 53 documentation requirements.
    - [Gebru et al., *Datasheets for Datasets* (2021)](https://arxiv.org/abs/1803.09010) — introduced structured dataset documentation whose sections map almost directly onto the Act's training-data summary obligations.
    - [Bommasani et al., *On the Opportunities and Risks of Foundation Models* (2021)](https://arxiv.org/abs/2108.07258) — Stanford CRFM report that established the systemic-risk framing now embedded in GPAI regulatory categories.

    **Recent advances (2023–2026)**

    - [Luccioni et al., *Power Hungry Processing: Watts Driving the Cost of AI Deployment?* (2023)](https://arxiv.org/abs/2311.16863) — empirical methodology for measuring inference energy, directly relevant to the Annex XI energy-consumption documentation field.
    - [EU AI Act — Regulation (EU) 2024/1689, Official Journal](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng) — the full legislative text; Chapter V with Annexes XI–XII (GPAI), Annex III (high-risk categories), Article 50 (transparency) and Article 73 (serious incidents) are the primary engineering-relevant sections.
    - [GPAI Code of Practice, Final Version (July 2025)](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai) — the European Commission's endorsed voluntary compliance tool for GPAI providers; adopting it gives legal certainty under Articles 53 and 55.

    **Open-source & tools**

    - [EU AI Act Compliance Checker (European Commission)](https://ai-act-service-desk.ec.europa.eu/en/eu-ai-act-compliance-checker) — official beta tool to determine which obligations apply to a given AI system or GPAI model.
    - [microsoft/presidio](https://github.com/microsoft/presidio/) — open-source PII detection and anonymisation framework (text, images, structured data) widely used to satisfy data-minimisation obligations in audit logs and training pipelines.
    - [mlco2/codecarbon](https://github.com/mlco2/codecarbon) — measures GPU/CPU/RAM energy and estimates CO2eq for a training or inference run; the practical way to fill the Annex XI energy-consumption field with a measured rather than guessed number.
    - [contentauth/c2pa-rs](https://github.com/contentauth/c2pa-rs) — Rust implementation (plus `c2patool` and language bindings) of the C2PA Content Credentials standard for signing provenance manifests into generated media, one of the two layers used to meet the Article 50 machine-readable marking duty.
    - [sigstore/model-transparency](https://github.com/sigstore/model-transparency) — OpenSSF model-signing tooling that hashes and signs model artefacts so a downstream user can verify the weights match the documentation you published.

    **Go deeper**

    - [NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework) — the GOVERN/MAP/MEASURE/MANAGE framework; primary US voluntary standard and operational backbone for EU AI Act risk-management system requirements.
    - [NIST AI 600-1: Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) — 2024 extension of the AI RMF specifically for foundation models, covering hallucination, data privacy, IP, and homogenisation risks.
    - [Hugging Face Model Cards Documentation](https://huggingface.co/docs/hub/model-cards) — practical guide to the model card metadata spec used across the industry and referenced by EU AI Act compliance tooling.

## Further Reading

- Mitchell et al., "Model Cards for Model Reporting," FAccT 2019 — the foundational paper defining the model card format.
- Gebru et al., "Datasheets for Datasets," Communications of the ACM, 2021 — the datasheet methodology, now effectively codified into regulation.
- NIST AI Risk Management Framework (AI RMF 1.0), NIST AI 100-1, January 2023 — the primary US voluntary governance framework.
- NIST Artificial Intelligence 600-1, "Generative AI Profile," 2024 — extension of the AI RMF to foundation and generative models.
- ISO/IEC 42001:2023, "Information technology — Artificial intelligence — Management system" — the auditable AI management system standard.
- Regulation (EU) 2024/1689 (the EU AI Act) — the full legislative text; Chapter V (Articles 51–56) with Annexes XI and XII is the GPAI regime, Article 50 the transparency regime, Article 73 serious incidents.
- European AI Office, GPAI Code of Practice (final version, July 2025) and the Commission's Guidelines on the scope of the GPAI obligations (2025) — the operationalisation guidance for GPAI providers, including the indicative compute threshold for being a GPAI model at all; check the EU AI Office website for the current version.
- W3C TDM Reservation Protocol (TDMRep) Community Group Final Report — the machine-readable rights-reservation signals (`tdm-reservation`, `tdm-policy`, `/.well-known/tdmrep.json`) an EU-compliant crawler must honour.
- Luccioni et al., "Power Hungry Processing: Watts Driving the Cost of AI Deployment?" ACL 2023 — empirical energy measurement methodology relevant to the Annex XI energy field.
- Bommasani et al., "On the Opportunities and Risks of Foundation Models," Stanford CRFM 2021 — comprehensive analysis of systemic risks relevant to the systemic-risk regulatory category.

---

## Exercises

**1.** A DDoS attack knocks your inference API offline for four hours, and separately a user reports that your mental-health triage assistant gave biased advice that plausibly contributed to self-harm. Your SRE on-call wants to file both through the same incident channel. Explain why, under the EU AI Act, these are two different kinds of incident, and state which one (if any) triggers an Article 73 serious-incident notification.

??? note "Solution"

    The chapter's "Common pitfall" admonition draws the exact distinction. AI Act serious-incident reporting is triggered by *harm or potential harm to people*, not by service availability or a security breach per se.

    - The **DDoS attack** is a security/availability incident. It degrades the service but is not, by itself, an event that "resulted, or could have resulted, in death or serious harm to health," property damage, disruption of an essential service, or a fundamental-rights violation. It flows through the ordinary SRE/security incident-response path. (It could become an AI Act incident only if the outage itself caused one of those harms — e.g. an essential service failing.)

    - The **biased mental-health advice that plausibly contributed to self-harm** is exactly an Article 73 *serious incident*: an event leading directly or indirectly to death or serious harm to a person's health. It must be notified to the market-surveillance authority of the Member State where the incident occurred, within 15 days of becoming aware (10 if a death occurred) — with an initial incomplete report if the investigation is not finished by then.

    The engineering consequence stated in the chapter is that you build *separate triage paths*: one keyed on availability/security signals, one keyed on harm-to-people signals. Merging them into a single channel risks either flooding the regulatory pipeline with irrelevant outages or, worse, burying a genuine harm event inside routine SRE noise and missing the notification deadline.

**2.** A team trains a dense transformer with $200$ billion parameters on $8$ trillion tokens. (a) Using the Chinchilla FLOP approximation from the chapter, compute the training compute. (b) Is the model a GPAI model with systemic risk? (c) How many *additional* tokens (beyond the $8$ T already used) would be needed to cross the $10^{25}$-FLOP threshold, holding parameter count fixed?

??? note "Solution"

    The chapter gives $C_{\text{train}} \approx 6 \cdot N \cdot D$.

    **(a)** With $N = 2 \times 10^{11}$ and $D = 8 \times 10^{12}$:

    $$
    C = 6 \times (2 \times 10^{11}) \times (8 \times 10^{12})
      = 6 \times 1.6 \times 10^{24}
      = 9.6 \times 10^{24} \text{ FLOPs}
    $$

    **(b)** $9.6 \times 10^{24} < 10^{25}$, so the model is **below** the systemic-risk threshold. It is still a GPAI model (self-supervised, broad generality) and therefore carries the Article 53 base obligations — technical documentation, training-data summary, copyright policy, downstream-deployer information — but it does *not* trigger the Article 55 systemic-risk obligations (adversarial testing, two-day incident reporting, cybersecurity, energy disclosure). It sits close to the bar, so the FlopTracker's 80% warning would already have fired.

    **(c)** Solve for the token count $D^{*}$ that hits the threshold with $N$ fixed:

    $$
    D^{*} = \frac{10^{25}}{6 \cdot N} = \frac{10^{25}}{6 \times 2 \times 10^{11}}
          = \frac{10^{25}}{1.2 \times 10^{12}}
          \approx 8.33 \times 10^{12} \text{ tokens}
    $$

    Additional tokens needed: $8.33 \times 10^{12} - 8 \times 10^{12} \approx 3.3 \times 10^{11}$, i.e. roughly **330 billion more tokens**. A modest extension of the run would tip the model across the line and pull in the full Article 55 regime.

**3.** The EU AI Act caps fines as the *higher* of a fixed euro amount or a percentage of global annual turnover: EUR 35 M / 7% for prohibited practices, EUR 15 M / 3% for other violations, EUR 7.5 M / 1.5% for supplying incorrect information. Compute the maximum fine for each violation category for (a) a large provider with EUR 2 billion global annual turnover, and (b) a startup with EUR 100 million turnover. Which company is bound by the fixed cap rather than the percentage, and for which categories?

??? note "Solution"

    For each category, take $\max(\text{fixed cap}, \ \text{percentage} \times \text{turnover})$.

    **(a) Turnover = EUR 2 billion ($2\times10^{9}$):**

    - Prohibited: $\max(35\text{M},\ 0.07 \times 2\text{B}) = \max(35\text{M},\ 140\text{M}) = \textbf{EUR 140 M}$.
    - Other violations: $\max(15\text{M},\ 0.03 \times 2\text{B}) = \max(15\text{M},\ 60\text{M}) = \textbf{EUR 60 M}$.
    - Incorrect information: $\max(7.5\text{M},\ 0.015 \times 2\text{B}) = \max(7.5\text{M},\ 30\text{M}) = \textbf{EUR 30 M}$.

    For the large provider the **percentage term dominates in every category** — the euro caps are irrelevant to it.

    **(b) Turnover = EUR 100 million ($1\times10^{8}$):**

    - Prohibited: $\max(35\text{M},\ 0.07 \times 100\text{M}) = \max(35\text{M},\ 7\text{M}) = \textbf{EUR 35 M}$ (fixed cap).
    - Other violations: $\max(15\text{M},\ 0.03 \times 100\text{M}) = \max(15\text{M},\ 3\text{M}) = \textbf{EUR 15 M}$ (fixed cap).
    - Incorrect information: $\max(7.5\text{M},\ 0.015 \times 100\text{M}) = \max(7.5\text{M},\ 1.5\text{M}) = \textbf{EUR 7.5 M}$ (fixed cap).

    The **startup is bound by the fixed euro caps in all three categories**, because 7% of its turnover (EUR 7 M) is smaller than the EUR 35 M fixed floor, and likewise for the other rows. The "higher of" rule is what makes the euro caps bite hardest on smaller firms while the percentage bites hardest on large ones — a prohibited-practice fine of EUR 35 M is 35% of the startup's turnover but only 1.75% of the large provider's.

**4.** The chapter's `write_log_record` note says to "include a chain hash for tamper evidence in high-assurance deployments." Implement that: modify the logging code so each record embeds the SHA-256 of the *previous* record's serialized JSON, forming a hash chain (a mini blockchain). Then write a verifier that walks a log file and returns the index of the first tampered record, or `None` if the chain is intact.

??? note "Solution"

    We add a `prev_hash` field to each record and derive it from the fully serialized previous line, so any edit to an earlier record invalidates every subsequent link. The genesis record chains from a fixed sentinel.

    ```python
    # chained_logger.py
    # Tamper-evident extension of eu_ai_act_logger.py using a SHA-256 hash chain.

    import hashlib
    import json

    GENESIS_HASH = "0" * 64   # sentinel prev_hash for the first record


    def _record_digest(line: str) -> str:
        """SHA-256 of one serialized JSON log line (the exact bytes written)."""
        return hashlib.sha256(line.encode("utf-8")).hexdigest()


    def write_chained_record(record_dict: dict, log_path: str) -> str:
        """
        Append `record_dict` to log_path, injecting `prev_hash` = digest of the
        last line already in the file (or GENESIS_HASH if empty).
        Returns the digest of the line just written (the next record's prev_hash).
        """
        prev_hash = GENESIS_HASH
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln]
            if lines:
                prev_hash = _record_digest(lines[-1])
        except FileNotFoundError:
            pass  # first write -> genesis

        record_dict = dict(record_dict)          # do not mutate caller's dict
        record_dict["prev_hash"] = prev_hash
        # Canonical serialization: sort keys so the digest is reproducible.
        line = json.dumps(record_dict, sort_keys=True)

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return _record_digest(line)


    def verify_chain(log_path: str):
        """
        Walk the log file and check that each record's prev_hash equals the
        digest of the previous line. Returns the 0-based index of the first
        record whose link is broken, or None if the chain is intact.
        """
        with open(log_path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln]

        expected_prev = GENESIS_HASH
        for i, line in enumerate(lines):
            rec = json.loads(line)
            if rec.get("prev_hash") != expected_prev:
                return i                     # this record's back-pointer is wrong
            expected_prev = _record_digest(line)
        return None
    ```

    Why it detects tampering: `verify_chain` recomputes `expected_prev` from the *actual bytes* of each stored line. If an attacker edits record $k$ (changing an `output_text`, say), the digest of line $k$ changes, so at index $k+1$ the stored `prev_hash` no longer matches the recomputed digest and the verifier returns $k+1$. Editing record $k$'s own `prev_hash` instead breaks the check at index $k$. To forge a clean chain the attacker would have to rewrite record $k$ *and every record after it* — and if the head digest is anchored externally (published, or in a write-once store as the chapter suggests), even that is detectable. Using `sort_keys=True` makes the serialization canonical so verification is deterministic regardless of dict insertion order.

    ```python
    # Smoke test
    if __name__ == "__main__":
        path = "/tmp/chain-demo.jsonl"
        open(path, "w").close()
        for i in range(3):
            write_chained_record({"event_id": i, "output_text": f"out-{i}"}, path)
        assert verify_chain(path) is None            # intact

        with open(path) as fh:
            lines = fh.read().splitlines()
        rec0 = json.loads(lines[0]); rec0["output_text"] = "TAMPERED"
        lines[0] = json.dumps(rec0, sort_keys=True)
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        assert verify_chain(path) == 1               # break detected at next link
        print("chain verification OK")
    ```

**5.** Extend the incident pipeline with deadline logic. Take the chapter's configured budgets: a systemic-risk incident (Art. 55(1)(c), day count from the GPAI Code of Practice) gets **2 days** to notify the EU AI Office; an Article 73 serious incident gets **15 days** to notify the national authority; `SIGNIFICANT` is internal-only (no external deadline) and `MINOR` needs no notification. Implement `notification_deadline_ms(report)` returning the absolute deadline timestamp (or `None` when no external notification is required), and `is_overdue(report, now_ms)` returning whether the deadline has passed without a notification having been sent.

??? note "Solution"

    We map each `IncidentSeverity` to a deadline measured from `detection_timestamp_ms`, reusing the enum and `AIIncidentReport` dataclass from the chapter's `incident_reporter.py`. `MINOR` and `SIGNIFICANT` carry no external clock, so the deadline is `None`.

    ```python
    # incident_deadlines.py
    # Deadline extension for incident_reporter.py.

    from incident_reporter import AIIncidentReport, IncidentSeverity

    _MS_PER_DAY = 24 * 60 * 60 * 1000

    # Days allowed for EXTERNAL notification, keyed by severity.
    _DEADLINE_DAYS = {
        IncidentSeverity.SYSTEMIC: 2,    # Art. 55(1)(c): EU AI Office
        IncidentSeverity.SERIOUS: 15,    # Art. 73: national authority
        # SIGNIFICANT and MINOR: no external deadline
    }


    def notification_deadline_ms(report: AIIncidentReport):
        """
        Absolute deadline (Unix epoch ms) by which an external notification
        must be sent, or None if this severity requires no external report.
        """
        days = _DEADLINE_DAYS.get(report.severity)
        if days is None:
            return None
        return report.detection_timestamp_ms + days * _MS_PER_DAY


    def is_overdue(report: AIIncidentReport, now_ms: float) -> bool:
        """
        True iff an external notification was required, the deadline has passed,
        and no notification has been sent (notification_timestamp_ms is None).
        """
        deadline = notification_deadline_ms(report)
        if deadline is None:
            return False                       # nothing was due
        if report.notification_timestamp_ms is not None:
            return False                       # already notified -> not overdue
        return now_ms > deadline
    ```

    Notes on the design. The deadline is anchored to `detection_timestamp_ms` (the moment of "becoming aware"), matching the Act's wording. A `SYSTEMIC` incident detected at day 0 must be reported within `2 * 86_400_000` ms; if `now_ms` exceeds that and `notification_timestamp_ms` is still `None`, `is_overdue` returns `True`, which is exactly the condition a monitor should page on. Once `notify_authority` sets `notification_timestamp_ms`, the incident is no longer overdue even after the deadline — the obligation was met. `SIGNIFICANT`/`MINOR` return `None`/`False` because the chapter classifies them as internal-only, so they never fire an external-deadline alert.

    ```python
    # Sanity check
    if __name__ == "__main__":
        r = AIIncidentReport(
            incident_id="x", detection_timestamp_ms=0.0,
            severity=IncidentSeverity.SYSTEMIC, description="d",
            affected_system="s", number_of_affected_persons=None,
            harm_category="health", corrective_measures_taken="", ongoing=True,
            detected_by="automated_monitor", assigned_to="team",
        )
        two_days = 2 * 24 * 60 * 60 * 1000
        assert notification_deadline_ms(r) == two_days
        assert is_overdue(r, now_ms=two_days + 1) is True     # missed it
        r.notification_timestamp_ms = two_days - 1000         # notified in time
        assert is_overdue(r, now_ms=two_days + 1) is False
        print("deadline logic OK")
    ```
