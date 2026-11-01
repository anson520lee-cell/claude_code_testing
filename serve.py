"""Start the research dashboard:  python serve.py  ->  http://127.0.0.1:8000

Options: --port 8080; --host 0.0.0.0 (reachable from other devices - needs a password first).
Password login: python serve.py --set-password   (python serve.py --clear-password removes it).
Remote access from your other devices: docs/REMOTE.md (Tailscale - private, free, no open ports).
The dashboard is read-only research; it never places orders.
"""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import sys

from src.daily.settings import load_terminal_settings
from src.platform_compat import ensure_utf8, redirect_output
from src.web import auth
from src.web.app import auth_file_for, create_app


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Swing research terminal dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", help="terminal settings file (default config/terminal.yaml)")
    parser.add_argument("--log", metavar="FILE", help="append all output to FILE (used by the Windows launcher)")
    parser.add_argument("--set-password", action="store_true", help="set the login password and exit")
    parser.add_argument("--clear-password", action="store_true", help="remove the login password and exit")
    parser.add_argument("--password-status", action="store_true", help="print whether a password is set and exit")
    args = parser.parse_args()
    settings = load_terminal_settings(args.config)
    auth_file = auth_file_for(settings)
    if args.set_password:
        sys.exit(set_password_interactive(auth_file))
    if args.clear_password:
        print("Password removed - the dashboard is open again." if auth.clear_password(auth_file)
              else "No password was set.")
        return
    if args.password_status:
        print("set" if auth.load(auth_file) else "not set")
        return
    if not is_loopback(args.host) and auth.load(auth_file) is None:
        sys.exit(f"Refusing to listen on {args.host} without a password: anyone who can reach this address could "
                 "edit the trade journal and alerts. Run  python serve.py --set-password  first.")
    redirect_output(args.log)
    import uvicorn
    # live_reload: pick up config/universe changes (monthly rebuild) without a restart
    app = create_app(settings, live_reload=True, config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def set_password_interactive(auth_file) -> int:  # noqa: ANN001
    print("Login password for the dashboard (stored only as a salted hash in data/auth.json).")
    first = getpass.getpass("New password: ")
    if first != getpass.getpass("Repeat it: "):
        print("The two entries differ - nothing changed.")
        return 1
    try:
        auth.set_password(auth_file, first)
    except ValueError as exc:
        print(f"{exc} Nothing changed.")
        return 1
    print("Password set. Every device must sign in once (it stays signed in for 90 days).")
    return 0


if __name__ == "__main__":
    ensure_utf8()
    main()
