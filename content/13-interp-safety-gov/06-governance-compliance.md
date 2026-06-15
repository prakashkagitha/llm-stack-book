# 13.6 AI Governance, Compliance & Regulation

Shipping a large language model (LLM) product is no longer just a systems engineering problem; it is a legal and organisational engineering problem too. In the span of a few years, AI regulation has graduated from voluntary guidance documents to binding statutes with civil penalties, mandatory incident reporting, and third-party audits. Engineers who understand only the model stack — and not the compliance layer on top of it — will ship systems that expose their employers to nine-figure fines or force costly re-architectures after launch.

This chapter gives you the engineer's view of that compliance layer: the EU AI Act (the world's most comprehensive AI law), the NIST AI Risk Management Framework (AI RMF), ISO/IEC 42001, and the practical artefacts — model cards, datasheets, eval reports, audit trails — that satisfy all of them. We also cover the specific obligations that fall on frontier model providers, the systemic-risk threshold, copyright and provenance mechanics, and serious-incident reporting pipelines that you actually have to build.

Related background: [AI Safety: Scalable Oversight, Dangerous-Capability Evals & Frontier Safety](../13-interp-safety-gov/05-ai-safety-oversight.html) covers the technical safety work that feeds governance artefacts. [Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html) details the eval methods referenced in model card sections. [Watermarking, Provenance & AI-Content Detection](../13-interp-safety-gov/04-watermarking-provenance.html) covers technical provenance methods. [Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html) handles the data-subject rights angle.

---

## The Regulatory Landscape in 2025–2026

Before diving into specifics, a mental map helps. Several regulatory streams are converging simultaneously.


{{fig:gov-regulatory-landscape-map}}


These three frameworks are complementary: the EU AI Act tells you *what* you must do and by when; the NIST AI RMF tells you *how* to run the governance process; ISO/IEC 42001 tells you *how to prove* to an auditor that you run that process consistently.

---

## The EU AI Act: Structure and Timeline

The EU AI Act (Regulation (EU) 2024/1689) entered into force on 1 August 2024. Its obligations phase in over 36 months:

| Date | Obligation active |
|---|---|
| 2 Feb 2025 | Prohibited AI practices banned (Article 5) |
| 2 Aug 2025 | GPAI model obligations; AI literacy duties |
| 2 Aug 2026 | High-risk application obligations; notified-body audits; fines apply |
| 2 Aug 2027 | High-risk embedded systems (Annex I) grace period ends |

Engineers need to care most about **2 Aug 2025** (GPAI model obligations — affects every frontier model provider) and **2 Aug 2026** (high-risk application rules — affects deployers building on those models).

### Risk Tiers

The Act classifies AI systems into four tiers:

1. **Prohibited** — Social scoring by public authorities, real-time biometric surveillance in public spaces, AI that exploits vulnerable groups. No compliance path; simply illegal in the EU.
2. **High-risk** — Listed in Annex III (biometric identification, critical infrastructure, employment decisions, essential services, law enforcement, migration, administration of justice, democratic processes). Also anything that is a safety component of a product under EU product safety law.
3. **Limited-risk** — Chatbots, deep-fakes, emotion recognition. Transparency obligations only (must disclose AI-generated content to users).
4. **Minimal-risk** — Spam filters, AI in video games. No mandatory obligations.

Most LLM applications land in **limited-risk** if deployed for open consumer use, but many enterprise applications (hiring tools, medical triage, credit scoring) tip into **high-risk**, triggering a heavy compliance programme.

---

## General-Purpose AI (GPAI) Model Obligations

Title VIII of the Act creates a distinct regime for "general-purpose AI models" (GPAI) — models trained on broad data at scale that can be adapted to many downstream tasks. Practically, this means every large pre-trained language model, including models released open-weight.

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

    This *exceeds* $10^{25}$, triggering systemic-risk obligations. In practice as of 2025, models from Google (Gemini Ultra), OpenAI (GPT-4 class), Anthropic (Claude 3 Opus class), and Meta (Llama 3 405B trained at scale) are in or near this territory. Most research-scale and open models sit below it.

### GPAI Obligations for All Providers (Article 53)

Every GPAI model provider, regardless of compute, must:

1. **Technical documentation** — Maintain up-to-date documentation covering architecture, training data, compute, evaluation results, known limitations, intended and foreseeable uses, and content filtering measures.
2. **Training-data summary** — Publish a "sufficiently detailed summary" of training data used. The Office of AI (European AI Office) publishes a template; it includes: data sources, languages covered, data selection methodology, filtering applied, personal data handling, and copyright measures.
3. **Copyright compliance** — Implement a policy to comply with EU copyright law, including the text-and-data mining exceptions in the 2019 Copyright Directive. Retain records to demonstrate compliance.
4. **Downstream deployer information** — Provide AI system providers who integrate the GPAI model with documentation and instructions sufficient to comply with their own obligations.

### GPAI Systemic-Risk Obligations (Article 55)

For models above $10^{25}$ FLOPs, four *additional* obligations apply:

1. **Adversarial testing (red-teaming)** — Perform model evaluations, including adversarial testing, to identify and mitigate systemic risks.
2. **Incident reporting** — Report serious incidents and possible corrective measures to the European AI Office within two days of becoming aware.
3. **Cybersecurity measures** — Protect the model and its infrastructure against adversarial attacks.
4. **Energy efficiency reporting** — Report training energy consumption (in MWh) and inferred operational energy when known.

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

  # Safety evaluations required for GPAI providers
  safety_evals:
    - name: "Dangerous capabilities (bio, chem, cyber, radiological)"
      methodology: "Internal red-team + third-party assessment"
      pass: true
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

Article 62 of the EU AI Act requires providers and deployers of high-risk AI systems to notify the relevant national market surveillance authority of *serious incidents* — defined as: incidents that resulted, or could have resulted, in death or serious harm to health; significant property damage; serious and irreversible disruption of essential services; or violations of EU law protecting fundamental rights.

For GPAI providers with systemic risk, Article 55(1)(b) additionally requires reporting *systemic-risk incidents* directly to the European AI Office within **two days** of first becoming aware.

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
    SERIOUS = "serious"          # Art. 62 notification to national authority (≤15 days)
    SYSTEMIC = "systemic"        # Art. 55(1)(b) notification to EU AI Office (≤2 days)


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
    Replace with the EU AI Office AISOG portal API when it becomes available.
    Deadline: SYSTEMIC = 2 days; SERIOUS = 15 days (national authority).
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

    Many teams conflate their general security-incident response process with AI-Act incident reporting. The key difference: AI-Act incidents are triggered by *harm or potential harm to people*, not by service outages or security breaches per se. A DDOS attack on your inference API is a security incident; a model that caused a user to self-harm following biased mental-health advice is an AI Act serious incident. Build separate triage paths.

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

The EU AI Act Article 53(1)(e) requires GPAI providers to publish an annually updated transparency report. Several voluntary frameworks (the Frontier Safety Framework from major labs, the Responsible Scaling Policy pattern) add further structure. Key sections in such a report:

1. **Model population summary** — All live models, versions, compute tier, and GPAI/systemic-risk classification.
2. **Incident log summary** — Aggregated statistics on serious-incident notifications (without PII).
3. **Red-team summary** — High-level results of adversarial evaluations since last report.
4. **Copyright and data-rights updates** — Changes to training-data composition, new opt-out compliance actions.
5. **Energy and compute disclosure** — Training and operational energy per Article 55(1)(d).

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

SYSTEMIC_RISK_THRESHOLD = 1e25   # EU AI Act Art. 51(1)(a)
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

---

## Copyright, Provenance, and the Chain of Custody

Chapter [Watermarking, Provenance & AI-Content Detection](../13-interp-safety-gov/04-watermarking-provenance.html) covers the technical watermarking side. Here we focus on the *legal* provenance chain.

### The EU Copyright Dimension

The EU Copyright in the Digital Single Market Directive (2019/790), Articles 3 and 4, create a text-and-data mining (TDM) exception that permits ML training on lawfully accessed content — unless the rightsholder has "expressly reserved" their rights (an opt-out, typically filed via `robots.txt` or a machine-readable TDM reservation attached to the work).

For LLM developers, this creates a duty to:

1. **Check opt-out signals** at crawl time (`robots.txt`; `X-Robots-Tag`; `tdm-reservation` metadata per the Rightscom TDM Reservation Protocol).
2. **Exclude opted-out sources** from training data.
3. **Maintain a rights register** documenting the basis for including each source.
4. **Retain evidence** of opt-out checks, preferably a signed record of the `robots.txt` as it existed at crawl time.

The following snippet shows how to check TDM reservation signals when building a crawler:

```python
# tdm_checker.py
# Check TDM reservation signals before including a URL's content in training data.
# Should be called as part of the data collection pipeline.

import re
import urllib.robotparser
from urllib.parse import urlparse


TDM_RESERVATION_HEADERS = {
    "tdm-reservation",       # Rightscom protocol header
    "x-tdm-reservation",
    "x-robots-tag",          # Google's extension; check for "tdm: none"
}


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

    # 1. Check TDM reservation headers
    for h in TDM_RESERVATION_HEADERS:
        if h in headers:
            val = headers[h].lower()
            if "tdm" in val or "reservation" in val or "none" in val:
                result["tdm_reservation_header_present"] = True
                result["should_exclude"] = True
                result["reason"] = f"TDM opt-out header: {h}={headers[h]}"
                return result

    # 2. Check robots.txt for AI training crawlers
    # Common user-agent strings used by AI training crawlers
    ai_crawlers = ["GPTBot", "CCBot", "ClaudeBot", "anthropic-ai", "Common Crawl"]
    parsed = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(robots_txt.splitlines())

    for crawler in ai_crawlers:
        if not rp.can_fetch(crawler, url):
            result["robots_txt_disallows_tdm"] = True
            result["should_exclude"] = True
            result["reason"] = f"robots.txt disallows {crawler}"
            return result

    # 3. Check for inline <meta name="robots" content="notdmcontent"> patterns
    # (for HTML pages; caller would pass page source)
    # Not shown here for brevity; check the Rightscom TDM spec.

    result["should_exclude"] = False
    result["reason"] = "No opt-out signal detected"
    return result
```

### Provenance Metadata in the Pretraining Pipeline

Every document that enters training should carry a **provenance record** — source URL, crawl timestamp, licence, opt-out check result — stored alongside the tokenised data. This is cheap at training time and invaluable when a regulator or rightsholder later asks "did this document go into your training data?"

See [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) for the broader data pipeline engineering, and [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) for deduplication methods that also help with copyright compliance.

---

!!! interview "Interview Corner"

    **Q:** A regulator asks your team to demonstrate that your 400 B parameter LLM complies with the EU AI Act. Walk through the obligations that apply and the artefacts you would produce.

    **A:** First, calculate training compute: with roughly $C = 6 \times 4 \times 10^{11} \times 1.5 \times 10^{13} \approx 3.6 \times 10^{25}$ FLOPs, the model crosses the $10^{25}$ systemic-risk threshold, so both base GPAI obligations (Art. 53) and systemic-risk obligations (Art. 55) apply.

    Base GPAI artefacts: (1) Technical documentation covering architecture, training data, compute, evaluations, and limitations; (2) a training-data summary published publicly, covering data sources, licence basis, and opt-out compliance; (3) a copyright policy; (4) downstream-deployer documentation.

    Systemic-risk additions: (1) Evidence of adversarial testing / red-teaming; (2) a two-day serious-incident reporting pipeline connected to the EU AI Office; (3) cybersecurity documentation; (4) energy-consumption disclosure.

    I would present: the model card YAML, the rights register CSV, the pre-deployment eval report JSON, the audit log schema, the incident-triage code, and the FlopTracker summary. I would also produce the EU AI Office registration record and confirm Art. 53 training-data summary has been published.

---

!!! key "Key Takeaways"

    - The EU AI Act phases in obligations over 2025–2027; the most operationally significant dates are **Aug 2025** (GPAI model duties) and **Aug 2026** (high-risk application enforcement with fines up to 7% of global turnover).
    - The **systemic-risk threshold** is $10^{25}$ FLOPs of training compute. Models below it still carry GPAI documentation and copyright obligations; models above it add adversarial testing, two-day incident reporting to the EU AI Office, and energy disclosure.
    - **Compliance artefacts** — model cards, training-data summaries, rights registers, eval reports, and audit logs — should be generated automatically from metadata captured during training and evaluation, not reconstructed after the fact.
    - **Audit logs** for high-risk deployments must include input references, output text, invoker identity, and timestamps; use append-only or immutable storage.
    - **Serious-incident reporting** is triggered by harm to people, not system outages; build a separate triage pipeline distinct from general SRE incident response.
    - The **NIST AI RMF** (GOVERN/MAP/MEASURE/MANAGE) and **ISO/IEC 42001** are complementary: the RMF gives the risk vocabulary and workflow; ISO 42001 gives the auditable management-system scaffold; the EU AI Act sets the legal floor.
    - **Copyright compliance** requires checking TDM opt-out signals at crawl time (`robots.txt`, TDM reservation headers), maintaining a rights register, and retaining timestamped evidence of those checks.
    - **Model cards** (Mitchell et al., 2019) and **datasheets for datasets** (Gebru et al., 2021) are the primary documentation artefacts; the EU AI Act training-data summary maps directly onto the datasheet structure.

---

!!! sota "State of the Art & Resources (2026)"
    AI governance and compliance has rapidly moved from voluntary guidance to binding law: the EU AI Act is now in full enforcement for GPAI models (Aug 2025) and high-risk applications (Aug 2026), while the NIST AI RMF and ISO/IEC 42001 have become the operational backbone that organisations use to satisfy those obligations. The resources below cover the foundational papers, the primary regulatory texts, and the open-source tooling engineers need to build compliant systems.

    **Foundational work**

    - [Mitchell et al., *Model Cards for Model Reporting* (2019)](https://arxiv.org/abs/1810.03993) — the seminal paper that defined the model card format now codified in EU AI Act Article 53 documentation requirements.
    - [Gebru et al., *Datasheets for Datasets* (2021)](https://arxiv.org/abs/1803.09010) — introduced structured dataset documentation whose sections map almost directly onto the Act's training-data summary obligations.
    - [Bommasani et al., *On the Opportunities and Risks of Foundation Models* (2021)](https://arxiv.org/abs/2108.07258) — Stanford CRFM report that established the systemic-risk framing now embedded in GPAI regulatory categories.

    **Recent advances (2023–2026)**

    - [Luccioni et al., *Power Hungry Processing: Watts Driving the Cost of AI Deployment?* (2023)](https://arxiv.org/abs/2311.16863) — empirical methodology for measuring inference energy, directly relevant to EU AI Act Article 55(1)(d) energy-disclosure obligations.
    - [EU AI Act — Regulation (EU) 2024/1689, Official Journal](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng) — the full legislative text; Title VIII (GPAI) and Annex III (high-risk categories) are the primary engineering-relevant sections.
    - [GPAI Code of Practice, Final Version (July 2025)](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai) — the European Commission's endorsed voluntary compliance tool for GPAI providers; adopting it gives legal certainty under Articles 53 and 55.

    **Open-source & tools**

    - [EU AI Act Compliance Checker (European Commission)](https://ai-act-service-desk.ec.europa.eu/en/eu-ai-act-compliance-checker) — official beta tool to determine which obligations apply to a given AI system or GPAI model.
    - [microsoft/presidio](https://github.com/microsoft/presidio/) — open-source PII detection and anonymisation framework (text, images, structured data) widely used to satisfy data-minimisation obligations in audit logs and training pipelines.

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
- Regulation (EU) 2024/1689 (the EU AI Act) — the full legislative text; Recitals 97–110 and Title VIII are the most relevant for GPAI.
- European AI Office, GPAI Code of Practice drafts (2025) — the operationalisation guidance for GPAI providers; check the EU AI Office website for the latest version.
- Luccioni et al., "Power Hungry Processing: Watts Driving the Cost of AI Deployment?" ACL 2023 — empirical energy measurement methodology relevant to Art. 55(1)(d) reporting.
- Bommasani et al., "On the Opportunities and Risks of Foundation Models," Stanford CRFM 2021 — comprehensive analysis of systemic risks relevant to the systemic-risk regulatory category.
