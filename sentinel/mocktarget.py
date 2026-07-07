"""An intentionally (in)secure local HTTP app — a target to test the DAST tools.

This is the equivalent of a tiny DVWA / OWASP Juice Shop: it exposes both
**vulnerable** and **secure** endpoints so the scanners in :mod:`sentinel.dast`
can be measured for precision *and* recall. It binds to loopback on an ephemeral
port and is used by the test suite and ``scripts/run_dast.py``.

⚠️ It is deliberately vulnerable — for local scanner testing only. Never deploy.
"""

from __future__ import annotations

import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

SQL_ERROR = "You have an error in your SQL syntax near"


class _Handler(BaseHTTPRequestHandler):
    # silence request logging
    def log_message(self, *a):  # noqa: D401
        pass

    def _send(self, code, body, headers=None, cookies=None):
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        for c in (cookies or []):
            self.send_header("Set-Cookie", c)
        body_b = body.encode()
        self.send_header("Content-Length", str(len(body_b)))
        self.end_headers()
        self.wfile.write(body_b)

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        q = parse_qs(u.query)

        # --- insecure landing page: missing security headers, weak cookie ----
        if path == "/":
            self._send(200, "<h1>MockShop</h1>", headers={"Server": "MockShop/1.2.3"},
                       cookies=["session=abc123; Path=/"])
        # --- fully hardened page: all headers + secure cookie (control) ------
        elif path == "/secure":
            self._send(200, "<h1>Secure</h1>", headers={
                "Content-Security-Policy": "default-src 'self'",
                "Strict-Transport-Security": "max-age=63072000",
                "X-Frame-Options": "DENY",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Permissions-Policy": "geolocation=()",
            }, cookies=["session=abc123; Path=/; Secure; HttpOnly; SameSite=Strict"])
        # --- reflected XSS (unescaped) vs escaped control --------------------
        elif path == "/search":
            term = q.get("q", [""])[0]
            self._send(200, f"<p>Results for {term}</p>")   # vulnerable: no escaping
        elif path == "/safe-search":
            term = q.get("q", [""])[0]
            self._send(200, f"<p>Results for {html.escape(term)}</p>")
        # --- error-based SQL injection --------------------------------------
        elif path == "/user":
            uid = q.get("id", [""])[0]
            if "'" in uid or "--" in uid:
                self._send(500, f"<pre>{SQL_ERROR} '{uid}'</pre>")
            else:
                self._send(200, f"<p>User {uid}</p>")
        # --- server-side template injection ---------------------------------
        elif path == "/render":
            name = q.get("name", [""])[0]
            if "{{7*7}}" in name or "${7*7}" in name:
                self._send(200, "<p>Hello 49</p>")          # template evaluated
            else:
                self._send(200, f"<p>Hello {name}</p>")
        # --- path traversal --------------------------------------------------
        elif path == "/file":
            p = q.get("path", [""])[0]
            if "etc/passwd" in p and ".." in p:
                self._send(200, "root:x:0:0:root:/root:/bin/bash")
            else:
                self._send(200, "file contents")
        # --- command injection ----------------------------------------------
        elif path == "/ping":
            hostp = q.get("host", [""])[0]
            marker = ""
            if ";" in hostp or "$(" in hostp or "`" in hostp:
                marker = "\nuid=0(root) gid=0(root)"      # simulated command exec
            self._send(200, f"PING {hostp}{marker}")
        # --- exposed sensitive files ----------------------------------------
        elif path in ("/.env", "/.git/config"):
            self._send(200, "SECRET_KEY=super-secret\nDB_PASSWORD=hunter2")
        # --- rate-limited API (for stress testing) --------------------------
        elif path == "/api":
            self._send(200, "{\"ok\": true}", headers={"Content-Type": "application/json"})
        else:
            self._send(404, "not found")


class MockTarget:
    """Context manager that runs the mock app on an ephemeral loopback port."""

    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
