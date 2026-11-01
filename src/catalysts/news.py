"""News: free RSS feeds, fetched automatically for every stock in the universe and for the market.

Sources (no API key, standard-library XML parsing):
    Yahoo Finance   https://feeds.finance.yahoo.com/rss/2.0/headline?s=<SYM>      per stock
    Nasdaq          https://www.nasdaq.com/feed/rssoutbound?symbol=<SYM>         per stock
    Google News     https://news.google.com/rss/search?q=<query>                 candidates + market topics
Items are de-duplicated by normalised title, kept for ``max_age_days``, and tagged by keyword rules:
earnings, guidance, analyst, M&A, offering (dilution), legal, FDA / clinical, contract, product,
insider, macro, big move. Sentiment is a keyword heuristic (positive / negative / neutral) - a reading
aid, not a model; it never changes the score (there is no historical news to back-test it on).
Every fetch is cached on disk for ``cache_minutes`` so the hourly job and page views stay light.
Politeness (these are free endpoints shared with the price data): at most 4 parallel requests, and a host
that answers 429 / 403 / 503 is left alone for an hour (recorded on disk so every process respects it).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) SwingTerminal/1.0 (research; RSS reader)"}
YAHOO = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
NASDAQ = "https://www.nasdaq.com/feed/rssoutbound?symbol={sym}"
GOOGLE = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
MARKET_QUERIES = {"Stock market": "stock market today", "Federal Reserve": "Federal Reserve interest rates",
                  "Semiconductors / AI": "semiconductor AI stocks", "Nasdaq / tech": "Nasdaq tech stocks",
                  "Economy": "US economy inflation jobs report"}

TAGS: list[tuple[str, str]] = [
    ("earnings", r"\b(earnings|quarterly results|q[1-4]\b|eps|revenue (beat|miss)|beats? estimates|misses? estimates|results)\b"),
    ("guidance", r"\b(guidance|outlook|forecast|raises (its )?(full[- ]year|fy)|cuts (its )?(full[- ]year|fy))\b"),
    ("analyst", r"\b(upgrade[sd]?|downgrade[sd]?|price target|initiates? coverage|overweight|underweight|outperform|underperform|buy rating|sell rating)\b"),
    ("M&A", r"\b(acquire[sd]?|acquisition|merger|to buy|takeover|buyout|deal to)\b"),
    ("offering", r"\b(offering|share sale|convertible|dilut\w*|private placement|at-the-market|atm program)\b"),
    ("legal", r"\b(lawsuit|sued|probe|investigation|sec charges|class action|settlement|antitrust|subpoena)\b"),
    ("FDA", r"\b(fda|clinical trial|phase [123i]+|approval|trial results|topline)\b"),
    ("contract", r"\b(contract|partnership|agreement|award(ed)?|order from|collaboration)\b"),
    ("product", r"\b(launch(es|ed)?|unveil\w*|new chip|new product|release[sd]?)\b"),
    ("insider", r"\b(insider|ceo (buys|sells)|director (buys|sells)|form 4|stake)\b"),
    ("macro", r"\b(fed|fomc|inflation|cpi|jobs report|tariff|rates|treasury|recession)\b"),
    ("big move", r"\b(soar\w*|surg\w*|jump\w*|plung\w*|tumbl\w*|crash\w*|sink\w*|slump\w*|skyrocket\w*|tank\w*|rall(y|ies|ied))\b"),
]
POSITIVE = r"\b(beat\w*|raise[sd]?|upgrade[sd]?|soar\w*|surg\w*|jump\w*|record|strong|approval|approved|wins?|awarded|rall(y|ies|ied)|outperform|buy rating|tops)\b"
NEGATIVE = r"\b(miss\w*|cut[s]?|downgrade[sd]?|plung\w*|tumbl\w*|crash\w*|sink\w*|slump\w*|weak|lawsuit|probe|investigation|recall|delay\w*|offering|dilut\w*|layoffs?|warn\w*|fall[s]?|fell|drop[s]?|halt\w*|reject\w*|underperform|sell rating|is down|slid\w*|doubts?|concerns?|fears?|worries|sell-?off)\b"


JUNK = re.compile(r"(options chain|stock quote|price chart|historical prices|stock price today|\bquote\b \||market cap \|)", re.I)
OFF_TOPIC = re.compile(r"\b(nse|bse|sensex|nifty|ftse|nikkei|kospi|hang seng|asx|tsx)\b", re.I)


def _usable(title: str, publisher: str | None) -> bool:
    """English, not a quote / options-chain page."""
    text = (title or "") + " " + (publisher or "")
    non_ascii = sum(1 for ch in text if ord(ch) > 0x2FFF)          # CJK, Hangul, etc.
    return non_ascii == 0 and not JUNK.search(title or "") and not OFF_TOPIC.search(title or "")


def _norm_title(t: str) -> str:
    t = re.sub(r"\s+-\s+[^-]{2,60}$", "", t or "")          # Google appends " - Publisher"
    return re.sub(r"[^a-z0-9 ]", "", t.lower())[:90]


def classify(title: str) -> dict[str, Any]:
    low = (title or "").lower()
    tags = [name for name, pat in TAGS if re.search(pat, low)]
    pos, neg = len(re.findall(POSITIVE, low)), len(re.findall(NEGATIVE, low))
    return {"tags": tags, "sentiment": "positive" if pos > neg else "negative" if neg > pos else "neutral"}


def parse_rss(xml_text: str, source: str) -> list[dict[str, Any]]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        pub = item.findtext("pubDate")
        try:
            when = parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else None
        except (TypeError, ValueError):
            when = None
        publisher = (item.findtext("source") or "").strip() or source
        if source == "Google News" and " - " in title:
            title, _, tail = title.rpartition(" - ")
            publisher = publisher if publisher != source else tail
        out.append({"title": title.strip(), "url": link, "published": when.strftime("%Y-%m-%dT%H:%M:%SZ") if when else None,
                    "publisher": publisher, "feed": source})
    return out


class NewsClient:
    def __init__(self, cache_dir: Path, getter: Callable[[str], str | None] | None = None, cache_minutes: int = 50,
                 max_age_days: int = 21, offline: bool = False) -> None:
        self.dir = Path(cache_dir) / "news"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.getter = getter or self._http
        self.cache_minutes, self.max_age_days, self.offline = cache_minutes, max_age_days, offline
        self.blocked_file = self.dir / "blocked_hosts.json"

    def _blocked(self) -> dict[str, float]:
        try:
            data = json.loads(self.blocked_file.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return {h: t for h, t in data.items() if t > time.time()}

    def _block(self, host: str, seconds: int = 3600) -> None:
        data = self._blocked()
        data[host] = time.time() + seconds
        try:
            self.blocked_file.write_text(json.dumps(data))
        except OSError:
            pass

    def _http(self, url: str) -> str | None:
        import requests
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        if host in self._blocked():
            return None
        try:
            r = requests.get(url, headers=UA, timeout=15)
        except Exception as exc:  # noqa: BLE001
            logger.debug("news fetch failed %s: %s", url, exc)
            return None
        if r.status_code in (403, 429, 503):
            logger.warning("news host %s answered %s - pausing it for an hour", host, r.status_code)
            self._block(host)
            return None
        return r.text if r.status_code == 200 else None

    def _cached(self, url: str, source: str) -> list[dict[str, Any]]:
        f = self.dir / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".json")
        if f.exists() and (self.offline or time.time() - f.stat().st_mtime < self.cache_minutes * 60):
            try:
                return json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
        fail = f.with_suffix(".fail")
        if self.offline or (fail.exists() and time.time() - fail.stat().st_mtime < 600):
            return json.loads(f.read_text()) if f.exists() else []   # a failed feed is retried after 10 minutes
        text = self.getter(url)
        if text is None:
            fail.touch()
            return json.loads(f.read_text()) if f.exists() else []
        items = parse_rss(text, source)
        f.write_text(json.dumps(items))
        fail.unlink(missing_ok=True)
        return items

    def _finish(self, items: list[dict[str, Any]], must_match: list[str] | None = None) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        seen, out = set(), []
        for it in sorted(items, key=lambda x: x.get("published") or "", reverse=True):
            key = _norm_title(it["title"])
            if not key or key in seen or not _usable(it["title"], it.get("publisher")):
                continue
            if it.get("published") and datetime.strptime(it["published"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) < cutoff:
                continue
            if must_match and it["feed"] == "Google News" and not any(m.lower() in it["title"].lower() for m in must_match if m):
                continue
            seen.add(key)
            out.append({**it, **classify(it["title"])})
        return out

    def symbol_news(self, symbol: str, name: str | None = None, google: bool = False) -> list[dict[str, Any]]:
        items = self._cached(YAHOO.format(sym=quote_plus(symbol)), "Yahoo Finance") + \
            self._cached(NASDAQ.format(sym=quote_plus(symbol)), "Nasdaq")
        short = re.split(r"[ ,.]", name or "")[0] if name else None
        if google:
            q = f'"{symbol}" stock' + (f' OR "{short}"' if short and len(short) > 3 else "")
            items += self._cached(GOOGLE.format(q=quote_plus(q)), "Google News")
        return self._finish(items, must_match=[symbol, short])

    def market_news(self) -> list[dict[str, Any]]:
        out = []
        for topic, q in MARKET_QUERIES.items():
            for it in self._cached(GOOGLE.format(q=quote_plus(q + " when:2d")), "Google News"):
                if not OFF_TOPIC.search(it["title"]):
                    out.append({**it, "topic": topic})
        return self._finish(out)[:80]

    def many(self, symbols: list[str], names: dict[str, str] | None = None, google_for: set[str] | None = None,
             workers: int = 4) -> dict[str, list[dict[str, Any]]]:
        names, google_for = names or {}, google_for or set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            res = pool.map(lambda s: self.symbol_news(s, names.get(s), s in google_for), symbols)
            return dict(zip(symbols, res))


def summarize(items: list[dict[str, Any]], now: datetime | None = None, days: int = 3) -> dict[str, Any]:
    """Recent-news digest for one stock: counts, tone, flags worth reading before a trade."""
    now = now or datetime.now(timezone.utc)
    recent = [x for x in items if x.get("published") and
              datetime.strptime(x["published"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) >= now - timedelta(days=days)]
    tags = sorted({t for x in recent for t in x["tags"]})
    neg = [x for x in recent if x["sentiment"] == "negative"]
    pos = [x for x in recent if x["sentiment"] == "positive"]
    negtags = {t for x in neg for t in x["tags"]}
    flags = [t for t in ("offering", "legal", "guidance") if t in negtags] + [t for t in ("M&A", "FDA") if t in tags]
    return {"n_total": len(items), "n_recent": len(recent), "days": days, "positive": len(pos), "negative": len(neg),
            "tags": tags, "flags": flags, "latest": items[:3]}


def save(conn, symbol: str, items: list[dict[str, Any]]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO news (symbol, url, title, publisher, feed, published, tags, sentiment, topic, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(symbol, x["url"], x["title"], x.get("publisher"), x.get("feed"), x.get("published"), ",".join(x["tags"]),
              x["sentiment"], x.get("topic"), now) for x in items])


def load(conn, symbol: str | None = None, limit: int = 300, since_days: int = 30, tag: str | None = None) -> list[dict[str, Any]]:
    since = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sql = "SELECT symbol, url, title, publisher, feed, published, tags, sentiment, topic FROM news WHERE COALESCE(published, fetched_at) >= ?"
    args: list[Any] = [since]
    if symbol:
        sql += " AND symbol = ?"
        args.append(symbol)
    if tag:
        sql += " AND (',' || tags || ',') LIKE ?"
        args.append(f"%,{tag},%")
    sql += " ORDER BY COALESCE(published, fetched_at) DESC LIMIT ?"
    args.append(limit)
    rows, seen = [], set()
    for r in conn.execute(sql, args).fetchall():
        key = (r[0], _norm_title(r[2]))           # the same story stored from two feeds (different URLs)
        if key in seen or not _usable(r[2], r[3]):
            continue
        seen.add(key)
        rows.append(r)
    cols = ["symbol", "url", "title", "publisher", "feed", "published", "tags", "sentiment", "topic"]
    return [{**dict(zip(cols, r)), "tags": [t for t in (r[6] or "").split(",") if t]} for r in rows]
