"""Writes the Chinese summary table (docs/SWING42.md) from reports/swing42/results.json."""

from __future__ import annotations

import json
import sys

RULES = {
    "1": "Minervini 趨勢模板 + 10日波幅/50日波幅 ≤ 0.4-0.55 + 量縮，收市破 20/40 日高 + 放量",
    "2": "40 日內升 60-90%，旗形回調 ≤ 20-30%，收市破旗頂",
    "3": "升市日成交量 > 前 10 日最大跌市日成交量，企 SMA10/50 之上、離 SMA10 ≤ 3-6%",
    "4": "25/40 日箱體深度 ≤ 12-18%，收市破箱頂 + 放量，企 200 日線上",
    "5": "130 日杯 (深 12-35/50%)，柄回調 ≤ 12% 破柄頂；cheat=杯中段 5 日窄幅後破頂",
    "6": "開市價 + k×昨日波幅 (k=0.2-0.6) 觸發買入，可加 NR7",
    "7": "連續 3 個週收市相差 ≤ 1-2%，之後 10 日內收市破 3 週最高",
    "8": "ADX>25-35 且上升，3 日內掂 EMA20，收市破昨日高",
    "9": "兩個擺動低點相差 ≤3-5% (或頭肩底)，之前跌 ≥20%，收市破頸線",
    "10": "跌穿 20/50/100 日低位後即日收返上去 (收喺 K 線上半)",
    "11": "大陽 (≥1.5-2.5 ATR) 破 20 日高之前最後一根陰燭 = OB，30 日內首次回踩買",
    "12": "三根 K 線缺口 (今低 > 前兩日高) ≥ 0.25-1 ATR，20 日內回踩缺口買",
    "13": "擺動點結構：CHoCH = 下跌結構中收市破最後擺動高；BOS = 上升結構中破高",
    "14": "先掃低再 CHoCH (breaker) / 冇掃低 (mitigation)，回踩被突破嘅擺動高點區買",
    "15": "BOS/CHoCH 後回調到該段 62-79% (或 50-62%) 區間買",
    "16": "40-60 日窄幅區 (≤20-30%) 喺下跌 ≥25% 之後；Spring = 跌穿區底收返入；SOS = 放量破區頂",
    "17": "兩個相近擺動低點 = 流動性池，掃穿後即日收返上去",
    "18": "曾企穩 AVWAP 5 日以上，回踩 AVWAP (錨: 52週低/跳空事件/年初/52週高) 收返上去",
    "19": "4 條 AVWAP 擠埋 ≤ 0.5-2 ATR，收市企上全部",
    "20": "60 日成交量分佈 POC 10 日內上移 ≥ 0.5-2 ATR，價喺 POC 上",
    "21": "收市破 VAH，而上面 3 格成交量 ≤ POC 嘅 20-35%",
    "22": "reject: 開市喺 VAL 下面收返入價值區；breakout: 放量收破 VAH",
    "23": "過去 60-250 日最大成交量 K 線嘅高位，由上面回踩守住",
    "24": "12 個月揀最似嘅 20 對，價差 > 2σ 開倉，回歸平倉",
    "25": "布林帶縮入 Keltner 通道 ≥3-6 日後放出，動量向上",
    "26": "RSI(2) < 2/5/10 (可加 200 日線上)，收市高過 SMA5 走",
    "27": "20 日 Z 分數 < -2/-2.5/-3 (可加 200 日線上)",
    "28": "5/10 日升 ≥25-40% 後第一根陰燭做空",
    "29": "開市跳空低過昨日低，升返昨日低位買",
    "30": "跳空 ≥5-12% + 2-3 倍量 + 收喺 K 線上半 (新聞/財報代理)",
    "30b": "真實財報：EPS surprise > 0/5/15% 且財報日反應 > 0/3/6%，之後一日開市買",
    "31": "由 52 週高跌 ≥40-60% 嘅股，一日升 ≥8-12% + 3 倍量 + 收近最高",
    "32": "每月期權到期週 / 之後一週 買或空指數",
    "33": "成交量 ≥2.5-4 倍但價格變動 ≤1-2%，企 50 日線上",
    "34": "SEC Form 4：≥3 個高管 30 日內買 / ≥2 個合共 ≥25 萬 / CEO/CFO 買 ≥10 萬",
    "35": "S&P 500 納入/剔除：生效前 5 日 / 生效後 20-60 日",
    "36": "冇交易清單，用 MNA / CSD ETF 旁證",
    "37": "財報前 3/5/10 日買，財報前最後一個收市賣",
    "38": "每月底揀 RS (3/6/9/12 月加權) 前 5-20%，可加近 52 週高",
    "38b": "9 個板塊 ETF 揀 3/6/12 月最強 1/3 個，月度換倉",
    "39": "龍頭跳空 ≥5%+3 倍量，買佢最相關嘅 3-5 隻未跟升同業",
    "40": "SPY 跌但 HYG/TLT/銅金比 唔跟跌、或 SPY 新低 VIX 唔新高 → 買 SPY",
    "41": "月尾最後 1/3 日開市買指數，持有 4/6 日",
    "42": "12 月中揀年初至今最差 5-20% 嘅股，持有 20-30 日",
}

ORDER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
         "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "30b", "31", "32", "33", "34", "35", "36", "37", "38",
         "38b", "39", "40", "41", "42"]


def fmt_edge(r, unit):
    e = r.get("edge")
    if e is None:
        return "-"
    return f"{e:+.2f}R" if unit == "R" else f"{e * 100:+.2f}%"


def fmt_mean(r, unit):
    e = r.get("mean")
    if e is None:
        return "-"
    return f"{e:+.2f}R" if unit == "R" else f"{e * 100:+.2f}%"


def row(k, x):
    if "IS" not in x:
        return f"| {k} | {x['zh']} | {RULES.get(k, '')} | - | - | - | - | - | **{x['verdict']}** |"
    unit = "R" if x["chosen"]["unit"].startswith("R") else "%"
    i, o = x["IS"], x["OOS"]
    prm = ", ".join(f"{a}={b}" for a, b in x["chosen"]["params"].items() if a != "off")
    f = x.get("filtered")
    filt = f"<br>加「{f['filter']}」: {fmt_edge(f['OOS'], unit)} (t {f['OOS'].get('t') or 0:.1f})" if f else ""
    g = x.get("grid_share_oos_pos")
    gs = f"{g * 100:.0f}%" if g is not None else "-"
    dm = ""
    if o.get("edge_dm") is not None and x.get("verdict_dm") not in (None, "-"):
        e = o["edge_dm"]
        dm = (f"{e:+.2f}R" if unit == "R" else f"{e * 100:+.2f}%") + f" (t {o.get('t_dm') or 0:.1f})"
    v = x["verdict"] + (f" ／ {x['verdict_dm']}" if x.get("verdict_dm") not in (None, "-") else "")
    return (f"| {k} | {x['zh']} | {prm}；{x['chosen'].get('exit_zh', x['chosen'].get('exit', ''))} | {o.get('n', 0)} | {fmt_mean(o, unit)} | "
            f"{fmt_edge(i, unit)} (t {i.get('t') or 0:.1f}) | **{fmt_edge(o, unit)} (t {o.get('t') or 0:.1f})**{filt} | "
            f"{dm or '-'} | {gs} | **{v}** |")


def main(path, out):
    r = json.load(open(path))
    lines = ["| # | 策略 | 揀中參數；出場 (2000-14 揀) | 樣本外筆數 | 樣本外每筆淨回報 | 樣本內 vs 隨機 (t) | 樣本外 vs 隨機 (t) | 樣本外 vs 同日隨機 (t) | 參數格樣本外為正 | 結論 (vs隨機 ／ vs同日) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ORDER:
        if k in r:
            lines.append(row(k, r[k]))
    open(out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
