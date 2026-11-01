"""Optional password login for the dashboard (needed before it is reachable from other devices).

Set a password with ``python serve.py --set-password`` (asks twice, never echoed). It is stored only as a
salted PBKDF2-SHA256 hash in ``data/auth.json`` next to the database (``data/`` is git-ignored); the file also
holds a random key that signs the login cookie. Remove it with ``python serve.py --clear-password``.

When the file exists every page and API call needs a valid login cookie, except:
  * ``/login`` (the form) and ``/static/*`` (CSS / JS / icons, no data);
  * requests made on the host itself straight to ``http://127.0.0.1:<port>`` (client is loopback, the Host
    header is 127.0.0.1 / localhost, and no proxy header is present). The launcher scripts use this; requests
    relayed by Tailscale Serve, Cloudflare Tunnel or any other reverse proxy carry forwarding headers and
    therefore must log in.

The cookie is HttpOnly, SameSite=Lax (blocks cross-site form posts) and Secure when the request came over
HTTPS. Five wrong passwords from one address within 15 minutes lock that address out for 15 minutes.
Changing or clearing the password signs every device out.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "swing_auth"
SESSION_DAYS = 90
ITERATIONS = 310_000
MAX_FAILURES = 5
LOCKOUT_SECONDS = 15 * 60
MIN_PASSWORD_LENGTH = 8
PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "forwarded", "x-real-ip",
                 "cf-connecting-ip", "tailscale-user-login", "tailscale-funnel-request")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
OPEN_PATHS = ("/login", "/logout", "/static/", "/favicon.ico")


# ------------------------------------------------------------------ password file
def hash_password(password: str, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(candidate.hex(), digest)
    except (ValueError, TypeError):
        return False


def set_password(file: Path, password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Use at least {MIN_PASSWORD_LENGTH} characters.")
    file.parent.mkdir(parents=True, exist_ok=True)
    data = {"password_hash": hash_password(password), "cookie_key": secrets.token_hex(32),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, file)


def clear_password(file: Path) -> bool:
    if file.exists():
        file.unlink()
        return True
    return False


def load(file: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("password_hash") or not data.get("cookie_key"):
        return None
    return data


# ------------------------------------------------------------------ cookie
def make_token(key: str, now: float | None = None, days: int = SESSION_DAYS) -> str:
    expires = int((now or time.time()) + days * 86400)
    payload = f"v1.{expires}"
    sig = hmac.new(bytes.fromhex(key), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def check_token(key: str, token: str | None, now: float | None = None) -> bool:
    if not token:
        return False
    try:
        version, expires, sig = token.split(".")
        if version != "v1" or int(expires) < (now or time.time()):
            return False
        expected = hmac.new(bytes.fromhex(key), f"{version}.{expires}".encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------------------ request helpers
def _header(scope_headers: list[tuple[bytes, bytes]], name: str) -> str | None:
    name_b = name.encode()
    for key, value in scope_headers:
        if key.lower() == name_b:
            return value.decode("latin-1")
    return None


def is_direct_local(client_host: str | None, headers: list[tuple[bytes, bytes]]) -> bool:
    """True only for a request made on this machine straight to the server (not relayed by a proxy)."""
    if client_host not in {"127.0.0.1", "::1"}:
        return False
    if any(_header(headers, h) is not None for h in PROXY_HEADERS):
        return False
    host = (_header(headers, "host") or "").strip().lower()
    hostname = urlsplit(f"//{host}").hostname if host else ""
    return hostname in LOCAL_HOSTS


def client_address(client_host: str | None, headers: list[tuple[bytes, bytes]]) -> str:
    forwarded = _header(headers, "x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return client_host or "unknown"


def is_https(scheme: str, headers: list[tuple[bytes, bytes]]) -> bool:
    proto = (_header(headers, "x-forwarded-proto") or "").split(",")[0].strip().lower()
    return scheme == "https" or proto == "https"


def safe_next(target: str | None) -> str:
    """Only same-site relative paths (no //host or scheme) - avoids an open redirect after login."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


class Lockout:
    def __init__(self) -> None:
        self.failures: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def blocked(self, address: str, now: float | None = None) -> bool:
        now = now or time.time()
        with self.lock:
            recent = [t for t in self.failures.get(address, []) if now - t < LOCKOUT_SECONDS]
            self.failures[address] = recent
            return len(recent) >= MAX_FAILURES

    def fail(self, address: str, now: float | None = None) -> None:
        with self.lock:
            self.failures.setdefault(address, []).append(now or time.time())

    def reset(self, address: str) -> None:
        with self.lock:
            self.failures.pop(address, None)


def login_page(next_path: str, error: str | None = None) -> str:
    message = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swing Terminal - Sign in</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:#000; color:#e5e7eb;
         font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  form {{ width:min(320px,calc(100vw - 32px)); padding:28px; border:1px solid #27272a; border-radius:12px;
         background:#0a0a0a; }}
  h1 {{ font-size:18px; margin:0 0 4px; }} p {{ color:#a1a1aa; margin:0 0 18px; font-size:13px; }}
  input {{ box-sizing:border-box; width:100%; padding:10px 12px; border-radius:8px; border:1px solid #3f3f46;
          background:#000; color:#fff; font-size:15px; }}
  button {{ margin-top:12px; width:100%; padding:10px; border:0; border-radius:8px; background:#22c55e;
           color:#000; font-weight:600; font-size:15px; cursor:pointer; }}
  .err {{ color:#f87171; margin:10px 0 0; }}
</style></head>
<body><form method="post" action="/login">
  <h1>Swing Terminal</h1><p>Research only. Sign in to continue.</p>
  <input type="hidden" name="next" value="{html.escape(next_path)}">
  <input type="password" name="password" placeholder="Password" autocomplete="current-password" autofocus required>
  <button type="submit">Sign in</button>{message}
</form></body></html>"""


# ------------------------------------------------------------------ FastAPI wiring
def install(app: Any, auth_file: Path) -> None:
    """Add the login routes and the guard middleware. The password file is re-read when it changes, so
    ``serve.py --set-password`` / ``--clear-password`` apply to a running server without a restart."""

    state: dict[str, Any] = {"stamp": None, "data": None}
    lockout = Lockout()

    def current() -> dict[str, Any] | None:
        try:
            st = auth_file.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        if stamp != state["stamp"]:
            state["data"] = load(auth_file) if stamp else None
            state["stamp"] = stamp
        return state["data"]

    @app.middleware("http")
    async def guard(request: Request, call_next):  # noqa: ANN202
        data = current()
        path_ = request.url.path
        if data is None or path_.startswith(OPEN_PATHS) or path_ == "/login":
            return await call_next(request)
        headers = request.scope.get("headers", [])
        client = request.client.host if request.client else None
        if is_direct_local(client, headers) or check_token(data["cookie_key"], request.cookies.get(COOKIE)):
            return await call_next(request)
        if path_.startswith("/api/") or request.method != "GET":
            return JSONResponse({"detail": "Login required"}, status_code=401)
        target = path_ + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(target, safe='/')}", status_code=303)

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    def login_form(next: str = "/") -> HTMLResponse:  # noqa: A002
        if current() is None:
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(login_page(safe_next(next)))

    @app.post("/login", include_in_schema=False)
    async def login_submit(request: Request):  # noqa: ANN202
        form = parse_qs((await request.body())[:4096].decode("utf-8", "replace"))
        password = (form.get("password") or [""])[0]
        data = current()
        target = safe_next((form.get("next") or ["/"])[0])
        if data is None:
            return RedirectResponse(target, status_code=303)
        headers = request.scope.get("headers", [])
        address = client_address(request.client.host if request.client else None, headers)
        if lockout.blocked(address):
            return HTMLResponse(login_page(target, "Too many attempts. Try again in 15 minutes."), status_code=429)
        if not verify_password(password, data["password_hash"]):
            lockout.fail(address)
            return HTMLResponse(login_page(target, "Wrong password."), status_code=401)
        lockout.reset(address)
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(COOKIE, make_token(data["cookie_key"]), max_age=SESSION_DAYS * 86400, httponly=True,
                            samesite="lax", secure=is_https(request.url.scheme, headers), path="/")
        return response

    @app.get("/logout", include_in_schema=False)
    def logout() -> RedirectResponse:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE, path="/")
        return response
