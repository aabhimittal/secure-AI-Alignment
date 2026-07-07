#!/usr/bin/env python3
"""One-command deploy of the SentinelLLM demo to a Hugging Face Space.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # or: export HF_TOKEN=hf_xxx

Usage:
    python deploy_space.py                       # -> <you>/sentinel-llm
    python deploy_space.py --repo you/my-space   # custom target

The Space runs the deterministic core, so it needs no secrets. To enable the
live `toxic-bert` bias audit, add an ``HF_TOKEN`` secret in the Space settings.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SPACE_README = """---
title: SentinelLLM Safety & Alignment Harness
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Jailbreak, bias & format guardrails for LLMs — live demo
---

# 🛡️ SentinelLLM

Interactive demo of three LLM trust-and-safety guardrails:

* **Jailbreak Guard** — de-obfuscates (base64/leetspeak/zero-width) and scores prompts.
* **Bias Probe** — counterfactual demographic fairness on any scorer/model.
* **Format Guard** — extract, repair & schema-validate JSON / XML.

Reproducible benchmarks and source: https://github.com/aabhimittal/secure-AI-Alignment
"""

# The Space gets gradio (and huggingface_hub) from the `sdk_version` in the
# frontmatter, so its requirements only list the one extra runtime dep.
SPACE_REQUIREMENTS = "jsonschema>=4.0\n"

# Only what the app needs at runtime.
UPLOAD = ["app.py",
          "sentinel/__init__.py", "sentinel/jailbreak.py", "sentinel/bias.py",
          "sentinel/formatctl.py", "sentinel/models.py", "sentinel/appsec.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None,
                    help="target space id, e.g. abhimittal/sentinel-llm")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        sys.exit("pip install huggingface_hub first")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    api = HfApi(token=token)
    who = whoami(token=token)
    user = who["name"]
    repo_id = args.repo or f"{user}/sentinel-llm"
    print(f"Deploying to Space: {repo_id}")

    api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                    exist_ok=True, token=token)

    # Space README (with the required YAML frontmatter).
    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space", token=token)

    # Space-specific requirements (gradio provided via sdk_version).
    api.upload_file(path_or_fileobj=SPACE_REQUIREMENTS.encode(),
                    path_in_repo="requirements.txt",
                    repo_id=repo_id, repo_type="space", token=token)

    for rel in UPLOAD:
        src = ROOT / rel
        if not src.exists():
            print(f"  skip (missing): {rel}")
            continue
        api.upload_file(path_or_fileobj=str(src), path_in_repo=rel,
                        repo_id=repo_id, repo_type="space", token=token)
        print(f"  uploaded {rel}")

    print(f"\n✅ Done → https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
