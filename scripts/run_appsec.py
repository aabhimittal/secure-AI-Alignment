#!/usr/bin/env python3
"""Reproducible SAST benchmark for the AppSec scanner.

Runs sentinel.appsec over a labeled vuln/safe corpus and reports precision /
recall / F1 (a vuln snippet must be flagged in the right category; a safe
snippet must yield no findings), plus a per-category breakdown and an
OWASP-category tally. Writes results/appsec.json and a chart.

    python scripts/run_appsec.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.appsec import AppSecScanner  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def main():
    scanner = AppSecScanner()
    cases = [json.loads(l) for l in (DATA / "appsec_cases.jsonl").open() if l.strip()]

    tp = fp = fn = tn = 0
    per_cat = defaultdict(lambda: [0, 0])   # category -> [detected, total] on vuln cases
    owasp = Counter()
    misses = []

    for c in cases:
        findings = scanner.scan(c["code"])
        cats = {f.category for f in findings}
        for f in findings:
            owasp[f.owasp] += 1

        if c["label"] == "vuln":
            hit = c["category"] in cats
            per_cat[c["category"]][1] += 1
            if hit:
                per_cat[c["category"]][0] += 1
                tp += 1
            else:
                fn += 1
                misses.append({"id": c["id"], "expected": c["category"], "got": sorted(cats)})
        else:  # safe
            if findings:
                fp += 1
                misses.append({"id": c["id"], "expected": "none",
                               "got": [f.rule_id for f in findings]})
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    out = {
        "n_cases": len(cases),
        "n_vuln": sum(1 for c in cases if c["label"] == "vuln"),
        "n_safe": sum(1 for c in cases if c["label"] == "safe"),
        "metrics": {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "recall_by_category": {k: round(v[0] / v[1], 4) for k, v in sorted(per_cat.items())},
        "owasp_findings_tally": dict(owasp.most_common()),
        "errors": misses,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "appsec.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"metrics": out["metrics"], "owasp": out["owasp_findings_tally"]}, indent=2))
    _chart(out)


def _chart(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    (RESULTS / "charts").mkdir(parents=True, exist_ok=True)
    tally = out["owasp_findings_tally"]
    labels = [k.split("-", 1)[0] for k in tally]   # e.g. A03:2021
    vals = list(tally.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels[::-1], vals[::-1], color="#76b900")
    ax.set_xlabel("Findings")
    ax.set_title(f"AppSec SAST — OWASP findings "
                 f"(P={out['metrics']['precision']:.2f} R={out['metrics']['recall']:.2f} "
                 f"F1={out['metrics']['f1']:.2f})")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS / "charts" / "appsec_owasp.png", dpi=130)
    print("chart -> results/charts/appsec_owasp.png")


if __name__ == "__main__":
    main()
