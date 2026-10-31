"""Export a self-contained, read-only preview of the dashboard (one HTML file, no server).

    python scripts/export_preview.py [--out preview/index.html] [--session YYYY-MM-DD]

All API answers the dashboard needs are captured from the real app (FastAPI test client) and
embedded in the page: snapshot, mini charts, full charts and analyst notes for every listed
candidate. Other tickers show a "not included in the preview" message.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.daily.settings import load_terminal_settings  # noqa: E402
from src.web.app import create_app  # noqa: E402

STATIC = ROOT / "src" / "web" / "static"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "preview" / "index.html"))
    ap.add_argument("--session")
    args = ap.parse_args()
    client = TestClient(create_app(load_terminal_settings()))
    get = lambda url: client.get(url).json()  # noqa: E731
    sessions = get("/api/sessions")["sessions"]
    session = args.session or sessions[0]
    snap = get(f"/api/snapshot?session={session}")
    cands = [c["symbol"] for side in ("long", "short") for c in snap["candidates"][side]]
    data = {
        "/api/sessions": {"sessions": [session]},
        "/api/snapshot": snap,
        "/api/i18n": get("/api/i18n"),
        "/api/methodology": get("/api/methodology"),
        "/api/runs": get("/api/runs"),
        "/api/regime/history": get("/api/regime/history"),
        "/api/run/status": {"running": False, "status": None, "log": []},
        "/api/minicharts": get(f"/api/minicharts?days=60&session={session}&symbols={','.join(cands)}"),
    }
    for sym in cands:
        data[f"/api/chart/{sym}"] = get(f"/api/chart/{sym}?days=160&session={session}")
        data[f"/api/candidate-history/{sym}"] = get(f"/api/candidate-history/{sym}")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    body = (STATIC / "index.html").read_text(encoding="utf-8")
    body = body[body.index("<body>") + 6: body.index('<script src="/static/app.js"></script>')]
    icon = "data:image/svg+xml;base64," + base64.b64encode((STATIC / "icon.svg").read_bytes()).decode()
    body = body.replace("/static/icon.svg", icon)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    shim = """
window.__PREVIEW_DATA__ = %s;
window.__PREVIEW__ = async (path) => {
  const D = window.__PREVIEW_DATA__, p = path.split("?")[0];
  if (p === "/api/minicharts") return D["/api/minicharts"];
  if (p.startsWith("/api/analyst/")) {
    const sym = decodeURIComponent(p.slice(13)).toUpperCase();
    const c = [...D["/api/snapshot"].candidates.long, ...D["/api/snapshot"].candidates.short].find((x) => x.symbol === sym);
    if (c && c.analyst) return { note: c.analyst };
    throw new Error("預覽版只包含候選股的分析師報告");
  }
  if (p.startsWith("/api/chart/") && !(p in D)) throw new Error("預覽版只包含候選股的圖表");
  if (p.startsWith("/api/candidate-history/") && !(p in D)) return { history: [] };
  if (p in D) return D[p];
  throw new Error("預覽版不提供此功能");
};
""" % payload
    html = f"""<title>Swing Terminal 預覽</title>
<meta name="description" content="高 Beta 波段研究終端的唯讀預覽（{session}）">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&family=Instrument+Serif:ital@1&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{css}
.topbar {{ top: env(safe-area-inset-top, 0px); }}
body::before {{ content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 40; background: var(--bg); }}
</style>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
{body}
<script>{shim}</script>
<script>{js}</script>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB  ({len(cands)} candidates, session {session})")


if __name__ == "__main__":
    main()
