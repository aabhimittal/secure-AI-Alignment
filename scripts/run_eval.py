#!/usr/bin/env python3
"""Reproducible evaluation runner for SentinelLLM.

Runs all three suites against the bundled real-world datasets and writes machine
readable results to ``results/*.json``. If ``matplotlib`` is installed it also
renders summary charts to ``results/charts/``.

    python scripts/run_eval.py

Everything is deterministic and offline — no API keys, no model downloads.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.jailbreak import JailbreakGuard, TRANSFORMS  # noqa: E402
from sentinel.bias import BiasProbe  # noqa: E402
from sentinel.formatctl import FormatController  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"


def _read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Suite 1 — jailbreak / adversarial detection
# --------------------------------------------------------------------------- #

def _prf(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return dict(precision=precision, recall=recall, f1=f1, accuracy=acc,
                tp=tp, fp=fp, fn=fn, tn=tn)


def run_jailbreak():
    guard = JailbreakGuard()
    harmful = _read_jsonl(DATA / "advbench_sample.jsonl")
    benign = _read_jsonl(DATA / "benign_sample.jsonl")

    # (1) Plain detection: harmful should be blocked/flagged, benign allowed.
    tp = fp = fn = tn = 0
    for row in harmful:
        v = guard.scan(row["text"])
        if v.verdict in ("block", "flag"):
            tp += 1
        else:
            fn += 1
    for row in benign:
        v = guard.scan(row["text"])
        if v.verdict in ("block", "flag"):
            fp += 1
        else:
            tn += 1
    plain = _prf(tp, fp, fn, tn)

    # (2) Evasion robustness: apply each obfuscation transform to harmful
    # prompts and measure recall (detection rate) per transform.
    evasion = {}
    for name, fn_transform in TRANSFORMS.items():
        detected = 0
        for row in harmful:
            v = guard.scan(fn_transform(row["text"]))
            if v.verdict in ("block", "flag"):
                detected += 1
        evasion[name] = round(detected / len(harmful), 4)

    # (3) Per-category recall on plain harmful prompts.
    per_cat = {}
    for row in harmful:
        cat = row.get("category", "other")
        v = guard.scan(row["text"])
        d = per_cat.setdefault(cat, [0, 0])
        d[1] += 1
        if v.verdict in ("block", "flag"):
            d[0] += 1
    per_category = {c: round(d[0] / d[1], 4) for c, d in per_cat.items()}

    return {
        "n_harmful": len(harmful),
        "n_benign": len(benign),
        "plain_detection": plain,
        "evasion_recall_by_transform": evasion,
        "recall_by_category": per_category,
    }


# --------------------------------------------------------------------------- #
# Suite 2 — counterfactual bias
# --------------------------------------------------------------------------- #

def run_bias():
    from sentinel.models import demonstrator_scorer

    templates = [t["template"] for t in _read_jsonl(DATA / "bias_templates.jsonl")]

    # Specificity: an identity-blind scorer must NOT be flagged as biased.
    fair_probe = BiasProbe(tolerance=0.15)
    fair_reports = fair_probe.probe_many(templates)
    fair_agg = fair_probe.aggregate(fair_reports)

    # Sensitivity: a scorer seeded with documented bias directions MUST be
    # flagged, and the probe should localise the least-favoured groups.
    biased_probe = BiasProbe(scorer=demonstrator_scorer, tolerance=0.15)
    biased_reports = biased_probe.probe_many(templates)
    biased_agg = biased_probe.aggregate(biased_reports)

    flagged = [r.to_dict() for r in biased_reports if not r.passed]
    # Which groups the probe most often names as least-favoured.
    from collections import Counter
    least = Counter(
        a.least_favoured for r in biased_reports for a in r.axis_results
        if a.disparity > 0.15
    )

    return {
        "n_templates": len(templates),
        "specificity_check": {
            "scorer": "identity-blind lexicon",
            "mean_disparity_by_axis": {k: round(v, 4) for k, v in fair_agg.items()},
            "flagged_templates": sum(1 for r in fair_reports if not r.passed),
            "note": "0 disparity expected — the probe raises no false bias signal",
        },
        "sensitivity_check": {
            "scorer": "demonstrator (documented bias directions; illustrative)",
            "mean_disparity_by_axis": {k: round(v, 4) for k, v in biased_agg.items()},
            "flagged_templates": sum(1 for r in biased_reports if not r.passed),
            "detection_rate": round(
                sum(1 for r in biased_reports if not r.passed) / len(biased_reports), 4),
            "most_flagged_groups": least.most_common(6),
        },
        "worst_offenders": sorted(
            flagged, key=lambda d: d["max_disparity"], reverse=True
        )[:5],
    }


# --------------------------------------------------------------------------- #
# Suite 3 — format conformance
# --------------------------------------------------------------------------- #

DEMO_JSON_SCHEMA = {"type": "object", "required": []}


def run_format():
    fc = FormatController()
    cases = _read_jsonl(DATA / "format_cases.jsonl")
    raw_ok = repaired_ok = still_bad = 0
    details = []
    for c in cases:
        if c["kind"] == "json":
            res = fc.process_json(c["raw"])
        else:
            res = fc.process_xml(c["raw"])
        raw_parses = c["expect_valid_raw"]
        if res.valid and not res.repaired:
            if raw_parses:
                raw_ok += 1
            else:
                repaired_ok += 1  # extracted without explicit repair flag
        elif res.valid and res.repaired:
            repaired_ok += 1
        else:
            still_bad += 1
        details.append({"id": c["id"], "kind": c["kind"], **res.to_dict()})

    n = len(cases)
    # Conformance if we can hand backend a valid object (with or without repair).
    conformant = sum(1 for d in details if d["valid"])
    naive = sum(1 for c in cases if c["expect_valid_raw"])
    return {
        "n_cases": n,
        "naive_parse_rate": round(naive / n, 4),
        "guarded_conformance_rate": round(conformant / n, 4),
        "recovered_by_repair": conformant - naive,
        "details": details,
    }


# --------------------------------------------------------------------------- #
# Charts (optional)
# --------------------------------------------------------------------------- #

def render_charts(jb, bias, fmt):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available — skipping charts")
        return

    CHARTS.mkdir(parents=True, exist_ok=True)
    ink = "#1f2933"
    accent = "#2f6df6"
    warn = "#e5484d"

    # Evasion recall by transform.
    ev = jb["evasion_recall_by_transform"]
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(ev.keys())
    vals = [ev[n] * 100 for n in names]
    colors = [accent if v >= 80 else warn for v in vals]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("Jailbreak detection under adversarial obfuscation")
    ax.set_ylim(0, 100)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9, color=ink)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS / "jailbreak_evasion.png", dpi=130)
    plt.close(fig)

    # Bias: specificity (fair scorer) vs sensitivity (biased demonstrator).
    fair = bias["specificity_check"]["mean_disparity_by_axis"]
    sens = bias["sensitivity_check"]["mean_disparity_by_axis"]
    ax_names = list(sens.keys())
    import numpy as np
    x = np.arange(len(ax_names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [fair[a] for a in ax_names], 0.4, label="identity-blind scorer", color=accent)
    ax.bar(x + 0.2, [sens[a] for a in ax_names], 0.4, label="biased demonstrator", color=warn)
    ax.axhline(0.15, ls="--", color=ink, lw=1, label="tolerance (0.15)")
    ax.set_xticks(x)
    ax.set_xticklabels(ax_names)
    ax.set_ylabel("Mean counterfactual disparity")
    ax.set_title("Bias probe: no false alarms, recovers injected bias")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(CHARTS / "bias_by_axis.png", dpi=130)
    plt.close(fig)

    # Format conformance before/after.
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Naive parse", "With Sentinel guard"]
    vals = [fmt["naive_parse_rate"] * 100, fmt["guarded_conformance_rate"] * 100]
    ax.bar(labels, vals, color=[warn, accent])
    ax.set_ylabel("Conformance rate (%)")
    ax.set_title("Structured-output conformance")
    ax.set_ylim(0, 100)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=11, color=ink)
    plt.tight_layout()
    fig.savefig(CHARTS / "format_conformance.png", dpi=130)
    plt.close(fig)
    print(f"charts written to {CHARTS}")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    jb = run_jailbreak()
    bias = run_bias()
    fmt = run_format()

    (RESULTS / "jailbreak.json").write_text(json.dumps(jb, indent=2))
    (RESULTS / "bias.json").write_text(json.dumps(bias, indent=2))
    (RESULTS / "format.json").write_text(json.dumps(fmt, indent=2))

    summary = {
        "jailbreak": {
            "plain_f1": round(jb["plain_detection"]["f1"], 4),
            "plain_recall": round(jb["plain_detection"]["recall"], 4),
            "plain_precision": round(jb["plain_detection"]["precision"], 4),
            "mean_evasion_recall": round(
                sum(jb["evasion_recall_by_transform"].values())
                / len(jb["evasion_recall_by_transform"]), 4),
        },
        "bias": {
            "specificity_flagged": bias["specificity_check"]["flagged_templates"],
            "sensitivity_detection_rate": bias["sensitivity_check"]["detection_rate"],
        },
        "format": {
            "naive_parse_rate": fmt["naive_parse_rate"],
            "guarded_conformance_rate": fmt["guarded_conformance_rate"],
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    render_charts(jb, bias, fmt)


if __name__ == "__main__":
    main()
