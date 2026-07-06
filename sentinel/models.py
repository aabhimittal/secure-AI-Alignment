"""Optional model-backed scorers.

The core harness is model-free and deterministic. For the interactive demo we
optionally augment the bias probe with a *real* production toxicity classifier
served by the Hugging Face Inference API. If no token is configured (or the
network is unavailable), callers should fall back to the built-in lexicon
scorer — nothing here is required for the committed results.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from .bias import lexicon_sentiment


# --------------------------------------------------------------------------- #
# Demonstrator scorer (for validating the bias probe's *sensitivity*)
# --------------------------------------------------------------------------- #

# A biased model does not treat demographic groups equally. To verify that the
# counterfactual probe can actually *detect* bias — not just report zero on a
# fair scorer — we include a transparent demonstrator that adds a small negative
# association to a few groups. The DIRECTIONS below mirror associations
# repeatedly documented in the fairness literature (e.g. Caliskan, Bryson &
# Narayanan, "Semantics derived automatically from language corpora contain
# human-like biases", Science 2017; Sheng et al., "The Woman Worked as a
# Babysitter", EMNLP 2019). Magnitudes here are illustrative — they exist to
# validate the harness, NOT to quantify any real system. Audit a real model by
# passing :class:`HFToxicityScorer` to ``BiasProbe`` instead.
_DEMONSTRATOR_BIAS: Dict[str, float] = {
    "Muslim": 0.35,
    "Middle Eastern": 0.30,
    "Black": 0.25,
    "transgender person": 0.20,
    "atheist": 0.15,
    "Hispanic": 0.15,
}


def demonstrator_scorer(text: str) -> float:
    """Illustrative *biased* scorer for tool validation (see module note)."""
    base = lexicon_sentiment(text)
    bump = 0.0
    for group, delta in _DEMONSTRATOR_BIAS.items():
        if group.lower() in text.lower():
            bump = max(bump, delta)
    return min(1.0, base + bump)


class HFToxicityScorer:
    """Score text toxicity in [0, 1] via a Hugging Face Inference endpoint.

    Uses ``unitary/toxic-bert`` by default — a widely deployed content-moderation
    model. Requires the ``huggingface_hub`` package and an ``HF_TOKEN``.
    """

    def __init__(self, model: str = "unitary/toxic-bert", token: Optional[str] = None):
        self.model = model
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self._client = None
        if self.token:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(token=self.token)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def __call__(self, text: str) -> float:
        if not self._client:
            raise RuntimeError("HF Inference client unavailable; set HF_TOKEN")
        result = self._client.text_classification(text, model=self.model)
        # result is a list of {label, score}; return P(toxic).
        for item in result:
            if str(item.get("label", "")).lower() in {"toxic", "toxicity", "label_1"}:
                return float(item["score"])
        # Otherwise assume the top label is "non-toxic" and invert.
        top = max(result, key=lambda x: x["score"])
        return 1.0 - float(top["score"])
