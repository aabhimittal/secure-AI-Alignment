#!/usr/bin/env python3
"""Reproducible DAST benchmark — spins up the local mock target and runs the
web-security scanner, the injection pentester and the stress tester against it.

    python scripts/run_dast.py

Everything runs on 127.0.0.1 (no external hosts). Writes results/dast.json
and a latency chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.mocktarget import MockTarget  # noqa: E402
from sentinel.dast import (  # noqa: E402
    WebSecurityScanner, InjectionTester, StressTester)

RESULTS = ROOT / "results"


def main():
    with MockTarget() as t:
        base = t.base_url

        # 1 · passive web security scan (vulnerable landing page)
        web = WebSecurityScanner().scan(base)
        # control: the hardened endpoint should have 0 *page-level* (header/cookie)
        # findings — host-level checks (exposed files, server banner) still apply.
        web_secure = WebSecurityScanner().scan(base + "/secure")
        page_level = {"security_headers", "cookie"}
        secure_page_findings = [f for f in web_secure if f.category in page_level]

        # 2 · active injection pentest across vulnerable + safe endpoints
        it = InjectionTester()
        pentest = {
            "xss@/search":            it.test(base, "/search", "q", ["xss"]),
            "xss@/safe-search":       it.test(base, "/safe-search", "q", ["xss"]),
            "sqli@/user":             it.test(base, "/user", "id", ["sql_injection"]),
            "ssti@/render":           it.test(base, "/render", "name", ["ssti"]),
            "traversal@/file":        it.test(base, "/file", "path", ["path_traversal"]),
            "cmdi@/ping":             it.test(base, "/ping", "host", ["command_injection"]),
        }
        confirmed = {k: [r.to_dict() for r in v if r.confirmed] for k, v in pentest.items()}

        # 3 · stress test the API endpoint
        stress = StressTester().run(base + "/api", requests=300, concurrency=25)

    n_confirmed = sum(len(v) for v in confirmed.values())
    out = {
        "target": "local mock app (127.0.0.1)",
        "web_security": {
            "vulnerable_page_findings": [f.to_dict() for f in web],
            "n_findings": len(web),
            "hardened_page_header_cookie_findings": len(secure_page_findings),  # expected 0
        },
        "pentest": {
            "endpoints_tested": len(pentest),
            "confirmed_vulnerabilities": n_confirmed,
            "confirmed": {k: v for k, v in confirmed.items() if v},
            "safe_endpoint_confirmed": len(confirmed.get("xss@/safe-search", [])),  # expected 0
        },
        "stress": stress.to_dict(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dast.json").write_text(json.dumps(out, indent=2))

    print(f"web-security: {len(web)} findings on vulnerable page; "
          f"{len(secure_page_findings)} header/cookie findings on hardened page (want 0)")
    print(f"pentest: {n_confirmed} confirmed PoCs across {len(pentest)} endpoints; "
          f"safe endpoint false-positives: {out['pentest']['safe_endpoint_confirmed']} (want 0)")
    print(f"stress: {stress.throughput_rps:.0f} rps, "
          f"p50={stress.latency_ms['p50']:.1f}ms p99={stress.latency_ms['p99']:.1f}ms, "
          f"{stress.ok} ok / {stress.errors} err")
    _chart(out)


def _chart(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    (RESULTS / "charts").mkdir(parents=True, exist_ok=True)
    lat = out["stress"]["latency_ms"]
    keys = ["p50", "p90", "p99", "max"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(keys, [lat[k] for k in keys], color="#76b900")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Stress test — {out['stress']['throughput_rps']:.0f} rps "
                 f"({out['stress']['requests']} reqs @ c={out['stress']['concurrency']})")
    for i, k in enumerate(keys):
        ax.text(i, lat[k], f"{lat[k]:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS / "charts" / "dast_latency.png", dpi=130)
    print("chart -> results/charts/dast_latency.png")


if __name__ == "__main__":
    main()
