#!/usr/bin/env python3
"""Deterministically triage audit/*.json into plan/worklist.json (Phase 2).

Groups all deficits by target chapter into per-chapter work orders. No LLM.
A work order is the unit of execution for the revise batches: one chapter,
all its deficits, processed by one pipeline item (avoids concurrent same-file edits).
"""
import json, glob, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Normalize agent target mislabels to the real chapter slug.
ALIASES = {
    "02-transformer/07-gpt-from-scratch": "02-transformer/07-build-gpt-from-scratch",
}

def norm(t):
    return ALIASES.get(t, t)

orders = collections.defaultdict(lambda: {"deficits": [], "units": set()})
grades = {}
notbuildable = {}
for f in sorted(glob.glob("audit/*.json")):
    d = json.load(open(f))
    uid = d["unit_id"]
    grades[uid] = d["grade"]
    if d.get("buildable_verdict") is False:
        notbuildable[uid] = d.get("buildable_reasoning", "")
    for x in d.get("deficits", []):
        t = norm(x.get("target", "UNKNOWN"))
        o = orders[t]
        o["deficits"].append({
            "unit": uid, "severity": x.get("severity"), "kind": x.get("kind"),
            "missing": x.get("missing"), "fix": x.get("fix"),
        })
        o["units"].add(uid)

SEV = {"critical": 0, "major": 1, "minor": 2}
work = []
for target, o in orders.items():
    defs = sorted(o["deficits"], key=lambda x: SEV.get(x["severity"], 9))
    nc = sum(1 for x in defs if x["severity"] == "critical")
    nm = sum(1 for x in defs if x["severity"] == "major")
    nmin = sum(1 for x in defs if x["severity"] == "minor")
    is_new = target.startswith("NEW:")
    # priority: chapters that block a not-buildable unit, then critical count, then major
    blocks_nb = bool(o["units"] & set(notbuildable))
    work.append({
        "target": target,
        "kind": "NEW" if is_new else "REVISE",
        "blocks_not_buildable": blocks_nb,
        "n_critical": nc, "n_major": nm, "n_minor": nmin, "n_total": len(defs),
        "units_affected": sorted(o["units"]),
        "has_correctness_bug": any(x["kind"] == "correctness-bug" for x in defs),
        "deficits": defs,
    })

work.sort(key=lambda w: (not w["blocks_not_buildable"], -w["n_critical"], -w["n_major"], -w["n_total"]))

plan = {
    "generated_from": "audit/*.json",
    "grade_histogram": dict(collections.Counter(grades.values())),
    "not_buildable_units": {u: r[:400] for u, r in notbuildable.items()},
    "totals": {
        "work_orders": len(work),
        "revise": sum(1 for w in work if w["kind"] == "REVISE"),
        "new": sum(1 for w in work if w["kind"] == "NEW"),
        "deficits": sum(w["n_total"] for w in work),
        "critical": sum(w["n_critical"] for w in work),
        "correctness_bugs": sum(1 for w in work for x in w["deficits"] if x["kind"] == "correctness-bug"),
    },
    "work_orders": work,
}
os.makedirs("plan", exist_ok=True)
json.dump(plan, open("plan/worklist.json", "w"), indent=2)

print(f"work orders: {len(work)}  ({plan['totals']['revise']} revise, {plan['totals']['new']} new)")
print(f"deficits: {plan['totals']['deficits']}  critical: {plan['totals']['critical']}  correctness-bugs: {plan['totals']['correctness_bugs']}")
print(f"\n{'#':>2}  {'C':>2}{'M':>3}{'m':>3}  {'nb':>2}  target")
print("-" * 70)
for i, w in enumerate(work, 1):
    nb = "!!" if w["blocks_not_buildable"] else ""
    print(f"{i:>2}  {w['n_critical']:>2}{w['n_major']:>3}{w['n_minor']:>3}  {nb:>2}  {w['target']}")
