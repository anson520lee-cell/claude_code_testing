import pandas as pd, yfinance as yf, sys, time
tab = pd.read_pickle("/tmp/rc/tab.pkl")
syms = sorted(set(tab.symbol) | {"SPY","QQQ","TQQQ","UPRO","SOXL","SSO","QLD","SMH","IWM","^VIX","^GSPC","^NDX","TLT","SHY","BIL","IWC","IWO","VTWO"})
syms = [s.replace(".", "-") for s in syms]
frames = {}
B = 50
for i in range(0, len(syms), B):
    chunk = syms[i:i+B]
    for attempt in range(3):
        try:
            d = yf.download(chunk, start="2003-06-01", auto_adjust=True, progress=False, group_by="ticker", threads=True)
            break
        except Exception as e:
            print("retry", e); time.sleep(5)
    for s in chunk:
        try:
            x = d[s].dropna(how="all")
            if len(x): frames[s] = x
        except Exception: pass
    print(i, len(frames), flush=True)
pd.to_pickle(frames, "/tmp/rc/prices.pkl")
print("done", len(frames), "of", len(syms))
