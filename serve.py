"""Start the research dashboard:  python serve.py  ->  http://127.0.0.1:8000

Options: --host 0.0.0.0 (reachable from other devices on your network), --port 8080.
The dashboard is read-only research; it never places orders.
"""

from __future__ import annotations

import argparse

import uvicorn

from src.daily.settings import load_terminal_settings
from src.platform_compat import ensure_utf8, redirect_output
from src.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Swing research terminal dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", help="terminal settings file (default config/terminal.yaml)")
    parser.add_argument("--log", metavar="FILE", help="append all output to FILE (used by the Windows launcher)")
    args = parser.parse_args()
    redirect_output(args.log)
    # live_reload: pick up config/universe changes (monthly rebuild) without a restart
    app = create_app(load_terminal_settings(args.config), live_reload=True, config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    ensure_utf8()
    main()
