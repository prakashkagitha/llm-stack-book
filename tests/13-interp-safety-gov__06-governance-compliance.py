"""
Runnability test for content/13-interp-safety-gov/06-governance-compliance.md

Tests the 4 heuristically CPU-runnable Python blocks from the chapter,
concatenated in chapter order (each block is standalone in this chapter --
none of them share names with each other, but they are kept in document
order per the task instructions):

    - block #0 (line ~129) -- eu_ai_act_logger.py: AIActLogRecord, make_log_record,
      write_log_record (Article 12 audit-log schema)
    - block #2 (line ~359) -- rights_register.py: DataSourceRecord, is_compliant,
      export_rights_register (Article 53(1)(c) copyright rights register)
    - block #3 (line ~454) -- incident_reporter.py: IncidentSeverity,
      AIIncidentReport, triage_incident, _infer_harm_category, notify_authority
      (Article 62 / 55(1)(b) serious-incident reporting pipeline)
    - block #5 (line ~700) -- flop_tracker.py: FlopTracker (systemic-risk
      1e25-FLOP threshold tracker)

Blocks #1 and #4 are non-Python (a YAML model card and a JSON eval report)
and are correctly skipped. Block #6 (tdm_checker.py) needs a real
robots.txt/HTTP-headers fetch pipeline around it to be meaningful and is
network-shaped by heuristic, so it is skipped per the task's SKIP list.

Deviations from the book's code (both are narrow "glue", not logic changes):
  - Block #0's `if __name__ == "__main__":` demo wrote to the hardcoded path
    "/var/log/ai-act-audit.jsonl", which requires root. The test writes to a
    tempfile instead; `write_log_record`'s own logic (open/append/json.dumps)
    is executed unchanged.
  - Block #3's `notify_authority` makes a real `smtplib.SMTP(...)` connection.
    Per the network-call rule, the test mocks `smtplib.SMTP` so the block's
    own logic (severity gating, JSON body construction, header assembly,
    invoking sendmail, updating notification bookkeeping fields) still runs
    for real, offline.
"""

import csv
import enum
import hashlib
import io
import json
import logging
import os
import smtplib
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from email.mime.text import MIMEText
from typing import List, Optional
from unittest.mock import MagicMock, patch


# =============================================================================
# Block #0 (line ~129): eu_ai_act_logger.py -- verbatim from the chapter.
# =============================================================================

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

    # Full output text -- retained for audit (encrypt at rest)
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
# Example usage (chapter's __main__ block, verbatim except the log path,
# which is redirected from "/var/log/ai-act-audit.jsonl" -- not writable in
# CI -- to a tempfile).
# ---------------------------------------------------------------------------
_tmp_dir = tempfile.mkdtemp(prefix="ai-act-audit-")
_audit_log_path = os.path.join(_tmp_dir, "ai-act-audit.jsonl")

record = make_log_record(
    invoker_id="user:pseudonym-7f3a",   # hashed / pseudonymised PII
    input_text="Is this loan application likely to be approved?",
    output_text="Based on the supplied financial data, the risk score ...",
    model_version="credit-risk-llm-v2.1.3",
    temperature=0.0,
    max_tokens=512,
    deployment_context="eu-high-risk:credit-scoring:annex-iii-b5",
)
write_log_record(record, _audit_log_path)
print(f"Logged event {record.event_id}")

# --- verify block #0 actually executed and produced a correct record -------
assert os.path.exists(_audit_log_path), "audit log file was not created"
with open(_audit_log_path, "r", encoding="utf-8") as _fh:
    _lines = _fh.readlines()
assert len(_lines) == 1, f"expected exactly one log line, got {len(_lines)}"
_parsed = json.loads(_lines[0])
assert _parsed["event_id"] == record.event_id
assert _parsed["invoker_id"] == "user:pseudonym-7f3a"
_expected_digest = hashlib.sha256(
    b"Is this loan application likely to be approved?"
).hexdigest()
assert _parsed["input_sha256"] == _expected_digest
assert _parsed["human_reviewer_id"] is None
print("[block #0] eu_ai_act_logger.py: OK")


# =============================================================================
# Block #2 (line ~359): rights_register.py -- verbatim from the chapter.
# =============================================================================

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


# ---------------------------------------------------------------------------
# Exercise block #2: build a tiny 3-row register covering the three
# is_compliant() branches (clean TDM exception, licence fallback, and an
# ignored opt-out violation), then export it to CSV.
# ---------------------------------------------------------------------------
_reg_records = [
    DataSourceRecord(
        source_id="CC-2023-11-EN",
        source_name="Common Crawl (filtered)",
        url="https://example.com/doc1",
        crawl_date="2023-11-01",
        licence="Unknown",
        rightsholder="unknown",
        text_data_mining_exception_applies=True,
        opt_out_detected=False,
        opt_out_respected=False,
        approximate_tokens=1000,
    ),
    DataSourceRecord(
        source_id="WIKI-2024-01",
        source_name="Wikipedia",
        url="https://example.com/doc2",
        crawl_date="2024-01-15",
        licence="CC-BY-SA-4.0",
        rightsholder="Wikimedia Foundation",
        text_data_mining_exception_applies=False,
        opt_out_detected=False,
        opt_out_respected=False,
        approximate_tokens=2000,
    ),
    DataSourceRecord(
        source_id="NEWS-2024-05",
        source_name="News archive",
        url="https://example.com/doc3",
        crawl_date="2024-05-20",
        licence="Proprietary",
        rightsholder="Example News Corp",
        text_data_mining_exception_applies=True,
        opt_out_detected=True,
        opt_out_respected=False,   # violation: opt-out was ignored
        approximate_tokens=500,
    ),
]

assert _reg_records[0].is_compliant() is True    # clean TDM exception
assert _reg_records[1].is_compliant() is True    # compliant licence
assert _reg_records[2].is_compliant() is False   # ignored opt-out

_csv_text = export_rights_register(_reg_records)
_csv_rows = list(csv.DictReader(io.StringIO(_csv_text)))
assert len(_csv_rows) == 3
assert _csv_rows[0]["compliant"] == "True"
assert _csv_rows[2]["compliant"] == "False"
print("[block #2] rights_register.py: OK")


# =============================================================================
# Block #3 (line ~454): incident_reporter.py -- verbatim from the chapter.
# =============================================================================

class IncidentSeverity(enum.Enum):
    """
    Severity classification mapping to regulatory reporting thresholds.
    """
    MINOR = "minor"              # Internal only; no external reporting required
    SIGNIFICANT = "significant"  # Log; 72-hour internal review required
    SERIOUS = "serious"          # Art. 62 notification to national authority (<=15 days)
    SYSTEMIC = "systemic"        # Art. 55(1)(b) notification to EU AI Office (<=2 days)


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


# ---------------------------------------------------------------------------
# Exercise block #3: triage a systemic-risk-level incident, then notify the
# authority. `smtplib.SMTP` is mocked (no real network call is permitted in
# this test); notify_authority's own logic -- body construction, header
# assembly, invoking the SMTP context manager's sendmail, and updating the
# report's bookkeeping fields -- all executes for real.
# ---------------------------------------------------------------------------
_incident = triage_incident(
    description="Model gave biased mental-health advice leading to user self-harm.",
    harm_indicators={
        "death": False,
        "serious_injury": True,
        "fundamental_rights_violation": False,
        "service_disruption": False,
        "property_damage": False,
        "broad_societal_impact": False,
        "affected_count": 1,
    },
    system_id="wellness-llm-v3.0.1",
)
assert _incident.severity == IncidentSeverity.SERIOUS
assert _incident.harm_category == "health"
assert _incident.number_of_affected_persons == 1

_incident_minor = triage_incident(
    description="Model gave a slightly inaccurate but harmless answer.",
    harm_indicators={},
    system_id="wellness-llm-v3.0.1",
)
assert _incident_minor.severity == IncidentSeverity.MINOR

_mock_smtp_instance = MagicMock()
_mock_smtp_instance.__enter__.return_value = _mock_smtp_instance
with patch("smtplib.SMTP", return_value=_mock_smtp_instance) as _mock_smtp_cls:
    notify_authority(_incident, smtp_host="smtp.internal.example.com",
                      authority_email="incidents@national-authority.example.eu")

_mock_smtp_cls.assert_called_once_with("smtp.internal.example.com")
assert _mock_smtp_instance.sendmail.called
_sendmail_args = _mock_smtp_instance.sendmail.call_args[0]
assert _sendmail_args[0] == "ai-governance@example.com"
assert _sendmail_args[1] == ["incidents@national-authority.example.eu"]
assert "SERIOUS" in _sendmail_args[2]
assert _incident.notified_authority == "incidents@national-authority.example.eu"
assert _incident.notification_timestamp_ms is not None
print("[block #3] incident_reporter.py: OK")


# =============================================================================
# Block #5 (line ~700): flop_tracker.py -- verbatim from the chapter.
# =============================================================================

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


# ---------------------------------------------------------------------------
# Exercise block #5: a tiny 400M-parameter model over a few large-token
# "batches" so the threshold is crossed within a handful of .step() calls
# (kept cheap: only python float arithmetic, no real tensors).
# ---------------------------------------------------------------------------
_tracker = FlopTracker(model_params=4e8)
assert _tracker.systemic_risk_reached() is False

# Below-threshold steps first.
_f1 = _tracker.step(tokens_in_batch=1_000_000)
assert _f1 == 6.0 * 4e8 * 1_000_000
assert _tracker.systemic_risk_reached() is False

# One large step to cross both the 80% warning fraction and the threshold
# itself: 6 * 4e8 * T >= 1e25  =>  T >= ~4.17e15.
_f2 = _tracker.step(tokens_in_batch=5_000_000_000_000_000)  # 5e15 tokens
assert _tracker.total_flops == _f1 + _f2
assert _tracker._warned is True
assert _tracker.systemic_risk_reached() is True

_summary = _tracker.summary()
assert _summary["systemic_risk_flag"] is True
assert _summary["fraction_of_threshold"] >= 1.0
assert _summary["total_flops"] == _tracker.total_flops
print("[block #5] flop_tracker.py: OK")


print("\nAll runnable blocks executed successfully.")
