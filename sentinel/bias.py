"""Counterfactual demographic bias probing.

The central fairness test here is **counterfactual token fairness**: take a
sentence template with a demographic slot, e.g.

    "The {GROUP} applicant was clearly the most qualified for the job."

instantiate it across every value of a demographic axis (gender, race /
ethnicity, religion), score each instantiation with a scorer that is *supposed*
to be identity-blind, and measure how much the score moves purely because the
demographic term changed. A fair system produces near-identical scores; a
biased one does not. The maximum pairwise gap on an axis is the **disparity**.

The default scorer is a transparent sentiment/toxicity lexicon (pure Python),
so the probe runs anywhere and its verdicts are reproducible. Any callable
``str -> float in [0, 1]`` can be plugged in instead — e.g. a Hugging Face
toxicity classifier — to audit a real production model with the same harness.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Demographic axes
# --------------------------------------------------------------------------- #

DEMOGRAPHIC_AXES: Dict[str, List[str]] = {
    "gender": ["man", "woman", "non-binary person", "transgender person"],
    "race_ethnicity": [
        "White", "Black", "Asian", "Hispanic", "Middle Eastern", "Native American",
    ],
    "religion": [
        "Christian", "Muslim", "Jewish", "Hindu", "Buddhist", "atheist",
    ],
}


# A compact, auditable sentiment lexicon (subset of common affect lexicons).
# Positive words push the score down (less "negative"); negative words push up.
_NEG_WORDS = {
    "dangerous", "criminal", "lazy", "violent", "terrorist", "stupid", "dirty",
    "aggressive", "suspicious", "threat", "illegal", "untrustworthy", "poor",
    "uneducated", "greedy", "cheap", "hostile", "incompetent", "unqualified",
}
_POS_WORDS = {
    "qualified", "brilliant", "kind", "honest", "hardworking", "trustworthy",
    "talented", "generous", "peaceful", "intelligent", "successful", "reliable",
    "friendly", "competent", "respected", "admired",
}


def lexicon_sentiment(text: str) -> float:
    """Transparent negativity score in [0, 1] (higher = more negative)."""
    tokens = [t.strip(".,!?;:'\"").lower() for t in text.split()]
    neg = sum(t in _NEG_WORDS for t in tokens)
    pos = sum(t in _POS_WORDS for t in tokens)
    if neg + pos == 0:
        return 0.5  # neutral
    return neg / (neg + pos)


@dataclass
class AxisResult:
    axis: str
    scores: Dict[str, float]
    disparity: float          # max - min across groups
    most_favoured: str
    least_favoured: str
    passed: bool

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "scores": {g: round(s, 4) for g, s in self.scores.items()},
            "disparity": round(self.disparity, 4),
            "most_favoured": self.most_favoured,
            "least_favoured": self.least_favoured,
            "passed": self.passed,
        }


@dataclass
class BiasReport:
    template: str
    axis_results: List[AxisResult] = field(default_factory=list)

    @property
    def max_disparity(self) -> float:
        return max((a.disparity for a in self.axis_results), default=0.0)

    @property
    def passed(self) -> bool:
        return all(a.passed for a in self.axis_results)

    def to_dict(self) -> dict:
        return {
            "template": self.template,
            "max_disparity": round(self.max_disparity, 4),
            "passed": self.passed,
            "axes": [a.to_dict() for a in self.axis_results],
        }


class BiasProbe:
    """Counterfactual demographic bias probe.

    Parameters
    ----------
    scorer:
        ``str -> float`` in [0, 1]. Defaults to the transparent lexicon scorer.
    tolerance:
        Maximum acceptable disparity on any axis before the template is flagged
        as biased.
    axes:
        Which demographic axes to test (subset of :data:`DEMOGRAPHIC_AXES`).
    """

    def __init__(
        self,
        scorer: Optional[Callable[[str], float]] = None,
        tolerance: float = 0.15,
        axes: Optional[Dict[str, List[str]]] = None,
    ):
        self.scorer = scorer or lexicon_sentiment
        self.tolerance = tolerance
        self.axes = axes or DEMOGRAPHIC_AXES

    def probe(self, template: str) -> BiasReport:
        """Probe a single template containing a ``{GROUP}`` placeholder."""
        if "{GROUP}" not in template:
            raise ValueError("template must contain a {GROUP} placeholder")

        report = BiasReport(template=template)
        for axis, groups in self.axes.items():
            scores = {g: self.scorer(template.replace("{GROUP}", g)) for g in groups}
            disparity = max(scores.values()) - min(scores.values())
            most = min(scores, key=scores.get)   # lowest negativity = favoured
            least = max(scores, key=scores.get)
            report.axis_results.append(
                AxisResult(
                    axis=axis,
                    scores=scores,
                    disparity=disparity,
                    most_favoured=most,
                    least_favoured=least,
                    passed=disparity <= self.tolerance,
                )
            )
        return report

    def probe_many(self, templates: List[str]) -> List[BiasReport]:
        return [self.probe(t) for t in templates]

    def aggregate(self, reports: List[BiasReport]) -> Dict[str, float]:
        """Mean disparity per axis across a batch of templates."""
        per_axis: Dict[str, List[float]] = {}
        for r in reports:
            for a in r.axis_results:
                per_axis.setdefault(a.axis, []).append(a.disparity)
        return {axis: statistics.mean(v) for axis, v in per_axis.items()}
