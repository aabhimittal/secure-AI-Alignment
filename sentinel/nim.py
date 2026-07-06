"""Live inference against NVIDIA NIM (build.nvidia.com) — no local weights.

The heavy model runs on NVIDIA's cloud; we only send OpenAI-compatible HTTP
requests, so this works on any CPU box (0 MB VRAM). Two thin wrappers:

* :class:`NIMClient`       — minimal OpenAI-compatible chat client (stdlib only).
* :class:`NIMSafetyGuard`  — wraps ``nvidia/llama-3.1-nemotron-safety-guard-8b-v3``,
  a production content-safety classifier, behind the same interface as the
  local :class:`sentinel.jailbreak.JailbreakGuard` so the two can be compared.

Set the API key via the ``NVIDIA_API_KEY`` environment variable. Nothing here
is imported by the offline core; it is opt-in for live benchmarking.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
SAFETY_GUARD_MODEL = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"


class NIMClient:
    """Minimal OpenAI-compatible chat client for NVIDIA NIM (stdlib only)."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = NIM_BASE_URL):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
        self.base_url = base_url.rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, model: str, messages: List[dict], *, max_tokens: int = 256,
             temperature: float = 0.0, timeout: int = 60) -> str:
        if not self.api_key:
            raise RuntimeError("no NVIDIA_API_KEY set")
        payload = json.dumps({
            "model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"]


@dataclass
class SafetyVerdict:
    prompt: str
    unsafe: bool
    categories: List[str] = field(default_factory=list)
    raw: str = ""

    @property
    def verdict(self) -> str:
        return "block" if self.unsafe else "allow"

    def to_dict(self) -> dict:
        return {"unsafe": self.unsafe, "categories": self.categories,
                "verdict": self.verdict}


class NIMSafetyGuard:
    """NVIDIA Nemotron Safety-Guard as a drop-in guard, mirroring JailbreakGuard."""

    def __init__(self, client: Optional[NIMClient] = None,
                 model: str = SAFETY_GUARD_MODEL):
        self.client = client or NIMClient()
        self.model = model

    @property
    def available(self) -> bool:
        return self.client.available

    def scan(self, prompt: str) -> SafetyVerdict:
        content = self.client.chat(
            self.model,
            [{"role": "user", "content": prompt}],
            max_tokens=100, temperature=0.0,
        )
        unsafe, cats = self._parse(content)
        return SafetyVerdict(prompt=prompt, unsafe=unsafe, categories=cats, raw=content.strip())

    @staticmethod
    def _parse(content: str) -> (bool, List[str]):
        content = content.strip()
        try:
            obj = json.loads(content)
            unsafe = str(obj.get("User Safety", "")).lower() == "unsafe"
            cats_raw = obj.get("Safety Categories", "")
            cats = [c.strip() for c in cats_raw.split(",")] if cats_raw else []
            return unsafe, cats
        except json.JSONDecodeError:
            # Fallback: substring check if the model didn't emit clean JSON.
            return ("unsafe" in content.lower()), []
