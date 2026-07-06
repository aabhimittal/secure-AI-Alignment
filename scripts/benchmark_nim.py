#!/usr/bin/env python3
"""Live head-to-head: SentinelLLM's local guard vs. NVIDIA's cloud safety model.

Runs both guards over the same AdvBench harmful + benign-control set and reports
precision / recall / F1 for each, plus their agreement. NVIDIA's
``llama-3.1-nemotron-safety-guard-8b-v3`` runs entirely on NVIDIA NIM — no local
weights, ~0 MB VRAM — so this works on a plain CPU box.

    export NVIDIA_API_KEY=nvapi-...
    python scripts/benchmark_nim.py

Writes results/nim_comparison.json (and a chart if matplotlib is present).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.jailbreak import JailbreakGuard  # noqa: E402
from sentinel.nim import NIMSafetyGuard  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"


def _read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
                accuracy=round(acc, 4), tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    guard = JailbreakGuard()
    nim = NIMSafetyGuard()
    if not nim.available:
        sys.exit("Set NVIDIA_API_KEY to run the live comparison.")

    harmful = _read_jsonl(DATA / "advbench_sample.jsonl")
    benign = _read_jsonl(DATA / "benign_sample.jsonl")
    items = [(r["text"], True) for r in harmful] + [(r["text"], False) for r in benign]

    local = dict(tp=0, fp=0, fn=0, tn=0)
    cloud = dict(tp=0, fp=0, fn=0, tn=0)
    agree = 0
    per_example = []

    for i, (text, is_harmful) in enumerate(items, 1):
        lv = guard.scan(text)
        l_block = lv.verdict in ("block", "flag")
        try:
            cv = nim.scan(text)
            c_block = cv.unsafe
            cats = cv.categories
        except Exception as e:  # network hiccup — record and continue
            c_block, cats = None, [f"error:{type(e).__name__}"]

        for store, pred in ((local, l_block), (cloud, c_block)):
            if pred is None:
                continue
            if is_harmful and pred:
                store["tp"] += 1
            elif is_harmful and not pred:
                store["fn"] += 1
            elif not is_harmful and pred:
                store["fp"] += 1
            else:
                store["tn"] += 1

        if c_block is not None and l_block == c_block:
            agree += 1
        per_example.append({
            "prompt": text[:90], "harmful": is_harmful,
            "local": l_block, "cloud": c_block, "cloud_categories": cats,
        })
        print(f"[{i:2d}/{len(items)}] local={'BLOCK' if l_block else 'allow':5} "
              f"cloud={'BLOCK' if c_block else ('allow' if c_block is not None else 'ERR')}"
              f"  {'harmful' if is_harmful else 'benign '}  {text[:50]}")
        time.sleep(0.15)  # be gentle on the free tier

    n_scored = sum(1 for e in per_example if e["cloud"] is not None)
    out = {
        "model_local": "SentinelLLM JailbreakGuard (stdlib)",
        "model_cloud": "nvidia/llama-3.1-nemotron-safety-guard-8b-v3 (NIM)",
        "n_examples": len(items),
        "n_scored_by_cloud": n_scored,
        "local_metrics": _prf(**local),
        "cloud_metrics": _prf(**cloud),
        "agreement_rate": round(agree / n_scored, 4) if n_scored else 0.0,
        "disagreements": [e for e in per_example
                          if e["cloud"] is not None and e["local"] != e["cloud"]],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "nim_comparison.json").write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print("local:", out["local_metrics"])
    print("cloud:", out["cloud_metrics"])
    print("agreement:", out["agreement_rate"])
    _chart(out)


def _chart(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    (RESULTS / "charts").mkdir(parents=True, exist_ok=True)
    metrics = ["precision", "recall", "f1", "accuracy"]
    local = [out["local_metrics"][m] for m in metrics]
    cloud = [out["cloud_metrics"][m] for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, local, 0.4, label="SentinelLLM (local, stdlib)", color="#2f6df6")
    ax.bar(x + 0.2, cloud, 0.4, label="NVIDIA Safety-Guard 8B (cloud)", color="#76b900")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Jailbreak detection: local guard vs. NVIDIA NIM safety model")
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(local, cloud)):
        ax.text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS / "charts" / "nim_comparison.png", dpi=130)
    print("chart -> results/charts/nim_comparison.png")


if __name__ == "__main__":
    main()
