"""SentinelLLM — a self-contained safety & alignment test harness for LLMs.

Three pillars, mirroring a production trust-and-safety pipeline:

* ``sentinel.jailbreak`` — adversarial / jailbreak prompt detection (red-team).
* ``sentinel.bias``      — counterfactual demographic bias probing.
* ``sentinel.formatctl`` — structured-output (JSON/XML) conformance + repair.

The detectors are pure-Python (standard library only) so that every number in
``results/`` is fully reproducible on any machine with ``python scripts/run_eval.py``
— no API keys, no model downloads, no network. Optional model-backed scoring
(Hugging Face Inference) is available for the interactive demo but never required.
"""

from .jailbreak import JailbreakGuard, JailbreakVerdict
from .bias import BiasProbe, BiasReport
from .formatctl import FormatController, FormatResult
from .appsec import AppSecScanner, Finding
from .dast import WebSecurityScanner, InjectionTester, StressTester

__version__ = "0.3.0"

__all__ = [
    "JailbreakGuard",
    "JailbreakVerdict",
    "BiasProbe",
    "BiasReport",
    "FormatController",
    "FormatResult",
    "AppSecScanner",
    "Finding",
    "WebSecurityScanner",
    "InjectionTester",
    "StressTester",
    "__version__",
]
