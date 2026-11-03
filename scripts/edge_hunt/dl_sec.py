"""SEC EDGAR: CIK map, XBRL companyfacts (point-in-time by 'filed' date) and 8-K filings (item 2.02 =
earnings release) for every ticker in our set. Generic User-Agent; <= 8 requests/second."""
import json, sys, time, threading, concurrent.futures as cf, pandas as pd, requests
RAW = "/home/user/research_data/raw"
UA = {"User-Agent": "research-script contact@example.com", "Accept-Encoding": "gzip, deflate"}
S = requests.Session(); S.headers.update(UA)
lock = threading.Lock(); last = [0.0]
def get(url):
    for a in range(5):
        with lock:
            wait = last[0] + 0.13 - time.time()
            if wait > 0: time.sleep(wait)
            last[0] = time.time()
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200: return r.json()
            if r.status_code == 404: return None
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    return None
m = get("https://www.sec.gov/files/company_tickers.json")
cik = {v["ticker"].replace(".", "-").upper(): int(v["cik_str"]) for v in m.values()}
json.dump(cik, open(f"{RAW}/sec_cik.json", "w"))
h = pd.read_csv(f"{RAW}/sp500_hist.csv", parse_dates=["date"])
tick = {t.strip().replace(".", "-") for row in h[h["date"] >= "2006-06-01"]["tickers"] for t in row.split(",")}
import yaml
cfg = yaml.safe_load(open("/home/user/claude_code_testing/config/terminal.yaml"))
tick |= set(map(str, (cfg.get("universe") or {}).keys()))
pairs = sorted({(t, cik[t]) for t in tick if t in cik})
print("tickers", len(tick), "with cik", len(pairs), flush=True)
CONCEPTS = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "GrossProfit",
            "NetIncomeLoss", "OperatingIncomeLoss", "Assets", "Liabilities", "StockholdersEquity",
            "NetCashProvidedByUsedInOperatingActivities", "CommonStockSharesOutstanding", "LongTermDebt",
            "LongTermDebtNoncurrent", "CashAndCashEquivalentsAtCarryingValue", "AssetsCurrent", "LiabilitiesCurrent",
            "ResearchAndDevelopmentExpense", "PaymentsForRepurchaseOfCommonStock", "EntityCommonStockSharesOutstanding",
            "EarningsPerShareDiluted", "CostOfRevenue", "InterestExpense"}
facts, eights = [], []
def one(p):
    t, c = p
    rows, ev = [], []
    f = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json")
    if f:
        for tax, cs in (f.get("facts") or {}).items():
            for name, body in cs.items():
                if name not in CONCEPTS: continue
                for unit, vals in body.get("units", {}).items():
                    for v in vals:
                        rows.append((t, c, name, unit, v.get("start"), v.get("end"), v.get("val"), v.get("filed"),
                                     v.get("form"), v.get("fp"), v.get("fy"), v.get("frame")))
    s = get(f"https://data.sec.gov/submissions/CIK{c:010d}.json")
    blocks = []
    if s:
        blocks.append(s.get("filings", {}).get("recent", {}))
        for extra in s.get("filings", {}).get("files", []):
            e = get(f"https://data.sec.gov/submissions/{extra['name']}")
            if e: blocks.append(e)
    for b in blocks:
        forms = b.get("form", [])
        for i, fm in enumerate(forms):
            if fm in ("8-K", "8-K/A", "10-Q", "10-K", "10-K405", "6-K", "20-F"):
                ev.append((t, c, fm, b["filingDate"][i], b.get("acceptanceDateTime", [""] * len(forms))[i],
                           b.get("items", [""] * len(forms))[i]))
    return rows, ev
n = 0
with cf.ThreadPoolExecutor(6) as ex:
    for rows, ev in ex.map(one, pairs):
        facts.extend(rows); eights.extend(ev); n += 1
        if n % 100 == 0: print(n, len(facts), len(eights), flush=True)
pd.DataFrame(facts, columns=["symbol", "cik", "concept", "unit", "start", "end", "val", "filed", "form", "fp", "fy", "frame"]
             ).to_parquet(f"{RAW}/sec_facts.parquet")
pd.DataFrame(eights, columns=["symbol", "cik", "form", "filing_date", "accepted", "items"]).to_parquet(f"{RAW}/sec_filings.parquet")
print("saved", len(facts), len(eights))
