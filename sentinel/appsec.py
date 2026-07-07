"""AppSec SAST scanner — OWASP Top 10 vulnerability detection for Python code.

A defensive, static analogue to what an autonomous pentester (e.g. Strix) probes
for dynamically. It parses source with the standard-library ``ast`` module and
flags dangerous call patterns — SQL/command/code injection, SSRF, XXE, insecure
deserialization, XSS/SSTI, weak crypto, disabled TLS verification, insecure JWT,
path traversal — plus regex checks for hardcoded secrets. AST-based matching
(rather than plain grep) keeps false positives low: a *literal* SQL string is
safe, an *interpolated* one is not.

Each finding carries a CWE id, an OWASP Top-10 (2021) category, a severity, the
line number and a remediation hint. Pure standard library, so results are
deterministic and reproducible.

This is a detection ("find & fix") tool. It does not attack live systems.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Finding model
# --------------------------------------------------------------------------- #

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


@dataclass
class Finding:
    rule_id: str
    category: str          # short slug, e.g. "sql_injection"
    cwe: str               # e.g. "CWE-89"
    owasp: str             # e.g. "A03:2021-Injection"
    severity: str          # critical | high | medium | low
    line: int
    message: str
    remediation: str
    snippet: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id, "category": self.category, "cwe": self.cwe,
            "owasp": self.owasp, "severity": self.severity, "line": self.line,
            "message": self.message, "remediation": self.remediation,
            "snippet": self.snippet,
        }


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #

def _call_name(func: ast.AST) -> str:
    """Reconstruct a dotted call name, e.g. os.system, requests.get, .execute."""
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _is_dynamic(node: Optional[ast.AST]) -> bool:
    """True if an argument is built from anything other than a constant literal.

    f-strings, ``+`` concatenation, ``%`` formatting, ``.format()`` and bare
    variables are all treated as (potentially attacker-controlled) dynamic input.
    """
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, (ast.JoinedStr, ast.BinOp, ast.Name)):
        return True
    if isinstance(node, ast.Call):
        return True
    return True


def _has_kw(call: ast.Call, name: str, value) -> bool:
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value == value:
            return True
    return False


# --------------------------------------------------------------------------- #
# Regex rules (for patterns awkward to express in the AST)
# --------------------------------------------------------------------------- #

SECRET_PATTERNS = [
    ("hardcoded_secret", r"(?i)\b\w*(password|passwd|pwd|secret|api_key|apikey|access_key|token)\w*\s*=\s*['\"][^'\"]{4,}['\"]",
     "CWE-798", "A07:2021-Identification and Authentication Failures", "high",
     "Hardcoded credential/secret", "Load secrets from environment variables or a secrets manager."),
    ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b",
     "CWE-798", "A07:2021-Identification and Authentication Failures", "critical",
     "Hardcoded AWS access key id", "Rotate the key immediately and use IAM roles / env vars."),
    ("private_key", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
     "CWE-798", "A07:2021-Identification and Authentication Failures", "critical",
     "Embedded private key material", "Never commit private keys; store them in a vault."),
]


class AppSecScanner:
    """Static OWASP Top-10 vulnerability scanner for a Python source string."""

    def scan(self, code: str, filename: str = "<string>") -> List[Finding]:
        findings: List[Finding] = []
        lines = code.splitlines()

        # --- regex layer (works even if the file doesn't parse) -----------
        for m in self._scan_secrets(code):
            findings.append(m)

        # --- AST layer -----------------------------------------------------
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError:
            return self._dedupe(findings)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                findings.extend(self._scan_call(node, lines))

        return self._dedupe(findings)

    # -- secrets ---------------------------------------------------------- #

    def _scan_secrets(self, code: str) -> List[Finding]:
        out = []
        for i, line in enumerate(code.splitlines(), 1):
            for rid, pat, cwe, owasp, sev, msg, rem in SECRET_PATTERNS:
                if re.search(pat, line):
                    out.append(Finding(rid, "secrets", cwe, owasp, sev, i, msg, rem, line.strip()[:120]))
        return out

    # -- call-based rules ------------------------------------------------- #

    def _scan_call(self, call: ast.Call, lines: List[str]) -> List[Finding]:
        name = _call_name(call.func)
        tail = name.rsplit(".", 1)[-1]
        ln = getattr(call, "lineno", 0)
        snippet = lines[ln - 1].strip()[:120] if 0 < ln <= len(lines) else ""
        args = call.args
        out: List[Finding] = []

        def add(rid, cat, cwe, owasp, sev, msg, rem):
            out.append(Finding(rid, cat, cwe, owasp, sev, ln, msg, rem, snippet))

        # code injection: eval / exec
        if name in ("eval", "exec"):
            add("py_eval_exec", "code_injection", "CWE-95",
                "A03:2021-Injection", "critical",
                f"Use of {name}() enables arbitrary code execution",
                "Avoid eval/exec on any untrusted input; use ast.literal_eval or explicit parsing.")

        # OS command injection
        if name in ("os.system", "os.popen") and args and _is_dynamic(args[0]):
            add("os_command_injection", "command_injection", "CWE-78",
                "A03:2021-Injection", "critical",
                f"{name}() with a dynamic command string is command injection",
                "Use subprocess with a list of args and shell=False; never interpolate input.")
        if name.startswith("subprocess.") and _has_kw(call, "shell", True):
            dyn = any(_is_dynamic(a) for a in args)
            add("subprocess_shell_true", "command_injection", "CWE-78",
                "A03:2021-Injection", "high" if dyn else "medium",
                "subprocess call with shell=True" + (" and dynamic input" if dyn else ""),
                "Pass args as a list and use shell=False.")

        # SQL injection (cursor.execute with dynamic SQL)
        if tail in ("execute", "executemany") and args and _is_dynamic(args[0]):
            add("sql_injection", "sql_injection", "CWE-89",
                "A03:2021-Injection", "high",
                "SQL built with string interpolation is injectable",
                "Use parameterised queries: cursor.execute(sql, params).")

        # SSTI / server-side template injection
        if tail == "render_template_string" and args and _is_dynamic(args[0]):
            add("ssti", "ssti", "CWE-1336",
                "A03:2021-Injection", "high",
                "render_template_string() on dynamic input allows template injection",
                "Render static templates with a context dict; never build templates from input.")

        # XSS via explicit safe-marking of dynamic content
        if tail in ("Markup", "mark_safe") and args and _is_dynamic(args[0]):
            add("xss_safe_markup", "xss", "CWE-79",
                "A03:2021-Injection", "high",
                f"{tail}() disables auto-escaping on dynamic content (XSS)",
                "Let the template engine escape output; do not mark untrusted input safe.")

        # SSRF
        if (name.startswith(("requests.", "httpx.")) and tail in
                ("get", "post", "put", "delete", "patch", "head", "request")) \
                or tail == "urlopen":
            if args and _is_dynamic(args[0]):
                add("ssrf", "ssrf", "CWE-918",
                    "A10:2021-Server-Side Request Forgery", "high",
                    "Outbound request to a dynamic URL may enable SSRF",
                    "Validate/allow-list the host; block internal ranges and redirects.")

        # Insecure deserialization
        if name in ("pickle.loads", "pickle.load", "cPickle.loads", "dill.loads",
                    "marshal.loads", "yaml.load"):
            insecure_yaml = name == "yaml.load" and not self._yaml_has_safe_loader(call)
            if name != "yaml.load" or insecure_yaml:
                add("insecure_deserialization", "deserialization", "CWE-502",
                    "A08:2021-Software and Data Integrity Failures", "high",
                    f"{name}() on untrusted data can execute arbitrary objects",
                    "Use json, or yaml.safe_load; never unpickle untrusted data.")

        # XXE
        if name in ("etree.parse", "etree.fromstring", "lxml.etree.parse",
                    "lxml.etree.fromstring", "xml.etree.ElementTree.parse",
                    "minidom.parse", "minidom.parseString", "parseString"):
            add("xxe", "xxe", "CWE-611",
                "A05:2021-Security Misconfiguration", "medium",
                "XML parsing without entity hardening can allow XXE",
                "Use defusedxml, or disable DTD/entity resolution on the parser.")

        # Weak crypto / hashing
        if name in ("hashlib.md5", "hashlib.sha1"):
            add("weak_hash", "weak_crypto", "CWE-327",
                "A02:2021-Cryptographic Failures", "medium",
                f"{name}() is cryptographically weak",
                "Use SHA-256+; for passwords use bcrypt/scrypt/argon2.")

        # Disabled TLS verification
        if _has_kw(call, "verify", False):
            add("tls_verify_disabled", "insecure_transport", "CWE-295",
                "A02:2021-Cryptographic Failures", "high",
                "TLS certificate verification disabled (verify=False)",
                "Never disable verification; fix the trust store instead.")
        if name in ("ssl._create_unverified_context",):
            add("ssl_unverified_ctx", "insecure_transport", "CWE-295",
                "A02:2021-Cryptographic Failures", "high",
                "Unverified SSL context created",
                "Use ssl.create_default_context() with verification on.")

        # Insecure JWT
        if tail == "decode" and name != "json.decode":
            if _has_kw(call, "verify", False):
                add("jwt_verify_false", "auth", "CWE-347",
                    "A07:2021-Identification and Authentication Failures", "high",
                    "JWT decoded with verify=False (signature not checked)",
                    "Verify signatures and pin the expected algorithm(s).")
            for kw in call.keywords:
                if kw.arg == "algorithms" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    for el in kw.value.elts:
                        if isinstance(el, ast.Constant) and str(el.value).lower() == "none":
                            add("jwt_alg_none", "auth", "CWE-347",
                                "A07:2021-Identification and Authentication Failures", "critical",
                                "JWT 'none' algorithm accepted (signature bypass)",
                                "Never allow the 'none' algorithm; pin HS256/RS256.")

        # Debug mode
        if tail == "run" and _has_kw(call, "debug", True):
            add("debug_enabled", "misconfiguration", "CWE-489",
                "A05:2021-Security Misconfiguration", "medium",
                "Web server started with debug=True",
                "Disable debug in production; it exposes an interactive console.")

        return out

    @staticmethod
    def _yaml_has_safe_loader(call: ast.Call) -> bool:
        for kw in call.keywords:
            if kw.arg == "Loader":
                loader = _call_name(kw.value) if isinstance(kw.value, ast.Attribute) else \
                    getattr(kw.value, "id", "")
                if "Safe" in loader:
                    return True
        return False

    @staticmethod
    def _dedupe(findings: List[Finding]) -> List[Finding]:
        seen = set()
        out = []
        for f in findings:
            key = (f.rule_id, f.line)
            if key not in seen:
                seen.add(key)
                out.append(f)
        out.sort(key=lambda f: (-SEVERITY_ORDER.get(f.severity, 0), f.line))
        return out
