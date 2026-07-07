#!/usr/bin/env python3
"""sentinel-scan — a Strix-inspired SAST CLI for Python code.

A lightweight, offline, defensive counterpart to `strix --target`: it scans a
file or directory for OWASP Top-10 patterns and, in CI mode, exits non-zero when
findings exist (like `strix -n`).

    python scripts/scan.py --target ./sentinel
    python scripts/scan.py --target app.py --min-severity high
    python scripts/scan.py --target ./ --diff-base origin/main    # changed files only
    python scripts/scan.py --target ./ --ci --format json         # CI gate

Exit code is 0 when clean, 1 when findings are reported (respecting --min-severity).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.appsec import AppSecScanner, SEVERITY_ORDER  # noqa: E402

C = {"critical": "\033[95m", "high": "\033[91m", "medium": "\033[93m",
     "low": "\033[96m", "reset": "\033[0m", "dim": "\033[2m"}


def _iter_py_files(target: Path, diff_base: str | None):
    if diff_base:
        try:
            out = subprocess.check_output(
                ["git", "diff", "--name-only", diff_base, "--"], cwd=ROOT, text=True)
            for rel in out.splitlines():
                p = ROOT / rel
                if p.suffix == ".py" and p.exists():
                    yield p
            return
        except subprocess.CalledProcessError:
            print(f"warning: git diff against {diff_base} failed; scanning all", file=sys.stderr)
    if target.is_file():
        yield target
    else:
        yield from target.rglob("*.py")


def main():
    ap = argparse.ArgumentParser(prog="sentinel-scan")
    ap.add_argument("--target", "-t", required=True, help="file or directory to scan")
    ap.add_argument("--min-severity", default="low",
                    choices=["low", "medium", "high", "critical"])
    ap.add_argument("--diff-base", help="only scan files changed vs this git ref")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--ci", "-n", action="store_true",
                    help="headless: exit 1 if any findings at/above --min-severity")
    args = ap.parse_args()

    scanner = AppSecScanner()
    threshold = SEVERITY_ORDER[args.min_severity]
    target = Path(args.target)
    all_findings = []

    for path in _iter_py_files(target, args.diff_base):
        try:
            code = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for f in scanner.scan(code, filename=str(path)):
            if SEVERITY_ORDER[f.severity] >= threshold:
                d = f.to_dict()
                d["file"] = str(path.relative_to(ROOT) if path.is_absolute() else path)
                all_findings.append(d)

    if args.format == "json":
        print(json.dumps({"findings": all_findings, "count": len(all_findings)}, indent=2))
    else:
        _print_text(all_findings)

    if args.ci and all_findings:
        sys.exit(1)
    sys.exit(0)


def _print_text(findings):
    if not findings:
        print("✅ sentinel-scan: no findings")
        return
    by_sev = {}
    for f in findings:
        by_sev.setdefault(f["severity"], 0)
        by_sev[f["severity"]] += 1
    print(f"🛡️  sentinel-scan: {len(findings)} finding(s)  "
          + "  ".join(f"{k}:{v}" for k, v in sorted(
              by_sev.items(), key=lambda kv: -SEVERITY_ORDER[kv[0]])))
    print()
    for f in findings:
        col = C.get(f["severity"], "")
        print(f"{col}[{f['severity'].upper():8}]{C['reset']} {f['owasp']}  {f['cwe']}")
        print(f"  {f['file']}:{f['line']}  {C['dim']}{f['message']}{C['reset']}")
        if f.get("snippet"):
            print(f"    {C['dim']}> {f['snippet']}{C['reset']}")
        print(f"    fix: {f['remediation']}")
        print()


if __name__ == "__main__":
    main()
