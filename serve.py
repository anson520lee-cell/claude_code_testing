"""Start the research dashboard:  python serve.py  ->  http://127.0.0.1:8000

Options: --host 0.0.0.0 (reachable from other devices on your network), --port 8080.
The dashboard is read-only research; it never places orders.
"""

from __future__ import annotations

import argparse

import uvicorn

from src.daily.settings import load_terminal_settings
from src.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Swing research terminal dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", help="terminal settings file (default config/terminal.yaml)")
    args = parser.parse_args()
    app = create_app(load_terminal_settings(args.config))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
