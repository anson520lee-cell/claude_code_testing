"""Point-in-time S&P 500 membership rebuilt from the Wikipedia change log (current list + every add/remove)."""
from __future__ import annotations
import io, re, sys
import pandas as pd, requests

UA = {"User-Agent": "research-bot/1.0 (anson520lee@gmail.com)"}


def _clean(x: str) -> str:
    x = re.sub(r"<ref.*?(</ref>|/>)", "", x, flags=re.S)
    x = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", x)
    return x.strip().lstrip("|").strip()


def fetch_changes(wiki_path: str | None = None) -> pd.DataFrame:
    if wiki_path:
        t = open(wiki_path).read()
    else:
        t = requests.get("https://en.wikipedia.org/w/index.php?title=Historical_components_of_the_S%26P_500&action=raw",
                         headers=UA, timeout=30).text
    body = t[t.find('id="changes"'):]
    body = body[: body.find("\n|}")]
    rows = []
    for blk in body.split("\n|-")[3:]:
        cells = []
        for line in blk.strip().split("\n"):
            if line.startswith("|"):
                parts = line[1:].split("||") if "||" in line else [line[1:]]
                cells += [p for p in parts]
        cells = [_clean(c) for c in cells]
        if len(cells) < 5:
            continue
        d = pd.to_datetime(cells[0].replace(",", ", ").replace("  ", " "), errors="coerce")
        if pd.isna(d):
            continue
        rows.append({"date": d, "add": cells[1], "add_name": cells[2], "rem": cells[3], "rem_name": cells[4]})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def membership_table(cur_symbols: list[str], changes: pd.DataFrame, since: str = "2004-01-01") -> pd.DataFrame:
    """Return long table (symbol, start, end) of S&P 500 membership intervals back to ``since``."""
    cur = set(cur_symbols)
    members = set(cur)
    iv: dict[str, list] = {s: [[pd.Timestamp("2262-01-01"), None]] for s in cur}  # [end, start]
    open_end: dict[str, pd.Timestamp] = {s: pd.Timestamp("2262-01-01") for s in cur}
    out = []
    for _, r in changes.sort_values("date", ascending=False).iterrows():
        if r["date"] < pd.Timestamp(since):
            break
        a, m = r["add"], r["rem"]
        if a and a in members:                     # undo an addition: it started on this date
            out.append((a, r["date"], open_end.pop(a))); members.discard(a)
        if m and m not in members:                 # undo a removal: it was a member until this date
            members.add(m); open_end[m] = r["date"]
    for s, e in open_end.items():
        out.append((s, pd.Timestamp(since), e))
    return pd.DataFrame(out, columns=["symbol", "start", "end"])


def members_on(tab: pd.DataFrame, day: pd.Timestamp) -> list[str]:
    m = tab[(tab.start <= day) & (tab.end > day)]
    return sorted(set(m.symbol))

if __name__ == "__main__":
    ch = fetch_changes("/tmp/rc/hist.wiki")
    print(len(ch), ch.date.min(), ch.date.max()); print(ch.head(3)); 
    cur = pd.read_csv("https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv").Symbol.tolist()
    tab = membership_table(cur, ch)
    for y in (2005, 2010, 2015, 2020, 2025):
        print(y, len(members_on(tab, pd.Timestamp(f"{y}-06-30"))))
    print(tab.symbol.nunique())
    tab.to_pickle("/tmp/rc/tab.pkl"); ch.to_pickle("/tmp/rc/ch.pkl")
