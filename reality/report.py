"""Build charts + the Chinese PDF report from the pickled study outputs."""
import pickle, json, sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False, "figure.dpi": 150})
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/project-files/reality-check"
os.makedirs(OUT, exist_ok=True)
L = lambda p: pickle.load(open(p, "rb"))
o1, o2, o3, o4 = (L(f"/tmp/rc/out{i}.pkl") for i in (1, 2, 3, 4))
eqs = {u: L(f"/tmp/rc/eq_{u}.pkl") for u in ("PIT", "SURV", "HIND")}
eq2 = L("/tmp/rc/eq2.pkl")
P = lambda d, k: d["periods"][k]
FULL, OOS, REC, AI = "全期 2005-26", "樣本外 2010-19", "近期 2020-26", "AI行情 2023-26"
m = o1["mom"]; lev = o2["lev"]; rv = o2["reversal"]; ml = o3["ml"]; sw = o3["sweep"]
pc = lambda x: f"{x*100:.0f}%" if abs(x) >= 0.095 else f"{x*100:.1f}%"
cm = lambda d, k: f"{pc(P(d,k)['cagr'])} / {pc(P(d,k)['mdd'])}"

# ------------------------------------------------ chart 1: headline bars
labels = ["動能前5檔\n用「今天的\nS&P500成分股」", "同一策略\n用「當年真正的\n成分股」", "從事後50大贏家\n隨機抽5檔\n(零技巧)", "TQQQ\n(3倍QQQ)\n2010-26真實",
          "高頻反轉\n紙上績效", "同策略\n扣10bps成本", "AI機器學習\n選股(誠實版)", "AI選股\n(含今日成分股偏差)", "SPY\n買入持有"]
vals = [P(m["SURV|5|10"], FULL)["cagr"], P(m["PIT|5|10"], FULL)["cagr"], np.mean([x[FULL]["cagr"] for x in m["HIND|5|10|rand"]]),
        P(lev["TQQQ 真實"], FULL)["cagr"], P(rv["紙上績效: 收盤進出、0成本"], FULL)["cagr"], P(rv["加 10bps 成本"], FULL)["cagr"],
        ml["AI選股 前20名 (成分股, 10bps)"]["full"]["cagr"], ml["AI選股 前20名 (含今日成分股偏差, 10bps)"]["full"]["cagr"], P(o1["SPY"] and {"periods": o1["SPY"]}, FULL)["cagr"]]
cols = ["#c0504d", "#4f81bd", "#c0504d", "#e69f00", "#c0504d", "#4f81bd", "#4f81bd", "#c0504d", "#7f7f7f"]
fig, ax = plt.subplots(figsize=(11, 4.6))
b = ax.bar(range(len(vals)), [v * 100 for v in vals], color=cols)
for i, v in enumerate(vals): ax.text(i, v * 100 + (1.2 if v >= 0 else -4), f"{v*100:.0f}%", ha="center", fontsize=10)
ax.axhline(90, color="k", ls="--", lw=1); ax.text(len(vals) - 0.5, 91.5, "「輕鬆90%」", ha="right", fontsize=10)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("年化報酬 CAGR (%)"); ax.set_ylim(-30, 100)
ax.set_title("同一批資料、同一天起跑:年化報酬(紅=含偏差/不計成本,藍=誠實條件)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(f"{OUT}/fig1_headline.png"); plt.close()

# ------------------------------------------------ chart 2: equity curves (log)
spy = pd.read_pickle("/tmp/rc/prices.pkl")["SPY"]["Close"].dropna(); spyr = spy.pct_change().loc["2005":]
fig, ax = plt.subplots(figsize=(11, 4.4))
for u, c, n in (("SURV", "#c0504d", "動能前10檔:用今天的S&P500成分股(倖存者)"), ("HIND", "#e69f00", "動能前10檔:只從事後50大贏家選"), ("PIT", "#4f81bd", "動能前10檔:當年真正的成分股"), ):
    r = eqs[u].loc["2005-06":]; ax.plot((1 + r).cumprod().index, (1 + r).cumprod().values, color=c, label=n)
r = spyr.loc["2005-06":]; ax.plot((1 + r).cumprod().index, (1 + r).cumprod().values, color="#555", label="SPY", lw=1.5)
ax.set_yscale("log"); ax.set_ylabel("淨值(2005-06 = 1,對數軸)"); ax.legend(fontsize=8, frameon=False, loc="upper left"); ax.grid(alpha=.25)
ax.set_title("同一個動能策略,只換「股票池怎麼來的」,年化差 20+ 個百分點、期末淨值差數十倍", fontsize=11); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_equity.png"); plt.close()

# ------------------------------------------------ chart 3: leveraged drawdowns
fig, ax = plt.subplots(figsize=(11, 3.8))
for k, c, n in (("QQQ 3x 買入持有(模擬)", "#c0504d", "3倍QQQ(模擬,2005起)"), ("TQQQ 真實", "#e69f00", "TQQQ 真實(2010起)"), ("QQQ 買入持有", "#4f81bd", "QQQ"), ("SPY 買入持有", "#555", "SPY")):
    r = eq2[k]; e = (1 + r).cumprod(); ax.plot(e.index, (e / e.cummax() - 1) * 100, color=c, label=n, lw=1.2)
ax.set_ylabel("距高點回撤 (%)"); ax.legend(fontsize=8, frameon=False, loc="lower left"); ax.grid(alpha=.25)
ax.set_title("3倍槓桿的代價:2008 年模擬 -95%,2022 年 TQQQ 實測 -82%", fontsize=11); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_leverage_dd.png"); plt.close()

# ------------------------------------------------ tables
def mom_row(name, key):
    d = m[key]; return f"<tr><td>{name}</td><td>{cm(d,FULL)}</td><td>{cm(d,OOS)}</td><td>{cm(d,REC)}</td><td>{cm(d,AI)}</td></tr>"
def rand_row(name, key):
    a = m[key]; f = lambda k, q: np.percentile([x[k]["cagr"] for x in a], q) if q else np.mean([x[k]["cagr"] for x in a])
    return f"<tr><td>{name}</td>" + "".join(f"<td>平均 {pc(f(k,0))} / 95分位 {pc(f(k,95))}</td>" for k in (FULL, OOS, REC, AI)) + "</tr>"
spyr_ = o1["SPY"]; spy_cell = lambda k: f"{pc(spyr_[k]['cagr'])} / {pc(spyr_[k]['mdd'])}"
t_mom = f"""<table><tr><th>股票池(都用同一個策略:12-1動能,月調倉,成本10bps)</th><th>全期 2005-26</th><th>2010-19</th><th>2020-26</th><th>2023-26</th></tr>
{mom_row('前5檔 · 今天的S&P500成分股(倖存者)','SURV|5|10')}{mom_row('前10檔 · 今天的S&P500成分股(倖存者)','SURV|10|10')}
{mom_row('前5檔 · 只從「事後50大贏家」選','HIND|5|10')}{mom_row('前5檔 · <b>當年真正的成分股</b>','PIT|5|10')}{mom_row('前10檔 · <b>當年真正的成分股</b>','PIT|10|10')}{mom_row('前20檔 · <b>當年真正的成分股</b>','PIT|20|10')}
<tr class=sep><td>對照:同股票池「隨機抽5檔」(每月重抽,40組)</td><td colspan=4></td></tr>
{rand_row('隨機5檔 · 今天成分股','SURV|5|10|rand')}{rand_row('隨機5檔 · 事後贏家','HIND|5|10|rand')}{rand_row('隨機5檔 · 當年成分股','PIT|5|10|rand')}
<tr class=sep><td>SPY 買入持有</td><td>{spy_cell(FULL)}</td><td>{spy_cell(OOS)}</td><td>{spy_cell(REC)}</td><td>{spy_cell(AI)}</td></tr></table>"""
def lev_row(name, k, note=""):
    d = lev[k]; return f"<tr><td>{name}</td><td>{cm(d,FULL)}</td><td>{cm(d,OOS)}</td><td>{cm(d,REC)}</td><td>{cm(d,AI)}</td><td>{note}</td></tr>"
ex = o2["lev_extra"]
t_lev = f"""<table><tr><th>標的(每格:年化 / 最大回撤)</th><th>全期*</th><th>2010-19</th><th>2020-26</th><th>2023-26</th><th>備註</th></tr>
{lev_row('QQQ 買入持有','QQQ 買入持有')}{lev_row('TQQQ 3倍QQQ(真實ETF,2010-02起)','TQQQ 真實','最長水下 %.1f 年' % ex['TQQQ 真實']['years_underwater'])}
{lev_row('SOXL 3倍半導體(真實ETF,2010-03起)','SOXL 真實','最長水下 %.1f 年' % ex['SOXL 真實']['years_underwater'])}
{lev_row('3倍QQQ(模擬,涵蓋2008)','QQQ 3x 買入持有(模擬)','2009-03 谷底 %s' % pc(ex['QQQ 3x 買入持有(模擬)']['mdd']))}
{lev_row('3倍QQQ + 200日線進出(模擬)','QQQ 3x + 200日線進出','進出 %d 次' % lev['QQQ 3x + 200日線進出']['switches'])}
{lev_row('3倍SMH + 200日線進出(模擬)','SMH 3x + 200日線進出','2023-26 是事後挑的視窗')}
<tr class=sep><td>SPY 買入持有</td><td>{spy_cell(FULL)}</td><td>{spy_cell(OOS)}</td><td>{spy_cell(REC)}</td><td>{spy_cell(AI)}</td><td></td></tr></table>
<p class=note>* 真實ETF的「全期」是從上市日起算(TQQQ/SOXL 約2010年),所以沒有包含2008。模擬版驗證:TQQQ 模擬 vs 真實 年化 {pc(o2['valid_TQQQ']['synthetic'])} vs {pc(o2['valid_TQQQ']['real'])},日報酬相關 {o2['valid_TQQQ']['corr']:.3f}。</p>"""
t_rev = "<table><tr><th>高頻短線反轉(每天買近2日最慘的10檔,持1天)</th><th>全期</th><th>2010-19</th><th>2020-26</th><th>年換手</th></tr>" + "".join(
    f"<tr><td>{k}</td><td>{pc(P(v,FULL)['cagr'])}</td><td>{pc(P(v,OOS)['cagr'])}</td><td>{pc(P(v,REC)['cagr'])}</td><td>{v['turn_per_year']:.0f}x</td></tr>" for k, v in rv.items()) + "</table>"
mlk = lambda k, p: ml[k]["periods"][p] if p in ml[k]["periods"] else None
t_ml = "<table><tr><th>機器學習選股(梯度提升樹,每年用過去資料重訓,只做長倉前20名,月調倉)</th><th>2011-26 全期</th><th>2010-19</th><th>2020-26</th><th>2023-26</th></tr>" + "".join(
    f"<tr><td>{k}</td><td>{pc(v['full']['cagr'])} / {pc(v['full']['mdd'])}</td>" + "".join(f"<td>{pc(v['periods'][p]['cagr'])} / {pc(v['periods'][p]['mdd'])}</td>" for p in (OOS, REC, AI)) + "</tr>"
    for k, v in ml.items()) + f"<tr class=sep><td>隨機抽20檔(同一批股票,25組)</td><td>平均 {pc(o3['ml_random_cagr']['mean'])} / 95分位 {pc(o3['ml_random_cagr']['p95'])}</td><td colspan=3></td></tr></table>"
roll = o3["rolling"]
t_roll = "<table><tr><th>動能前10檔,滾動窗口裡「年化≥90%」的比例</th><th>1年窗口</th><th>3年窗口</th><th>5年窗口</th></tr>" + "".join(
    f"<tr><td>{n}</td>" + "".join(f"<td>{roll[f'{u}|{y}y']['share_ge90']*100:.1f}%(中位數 {pc(roll[f'{u}|{y}y']['median'])})</td>" for y in (1, 3, 5)) + "</tr>"
    for u, n in (("SURV", "今天的S&P500成分股(倖存者)"), ("HIND", "事後贏家50檔"), ("PIT", "當年真正的成分股"))) + "</table>"
ps = pd.DataFrame(o3["persist"]).dropna()
fy = pd.DataFrame(o3["five_year"]); fy = fy[fy.n > 0]
proj = o4
pr_rand = lambda k, q, per: (np.percentile if q else np.mean)([x[per]["cagr"] for x in proj[k]], q) if q else np.mean([x[per]["cagr"] for x in proj[k]])
sv = o3["sweep"]
res = dict(m=m, lev=lev, rv=rv, ml=ml, sweep=sv, roll=roll, persist=o3["persist"], five=o3["five_year"], proj={k: v for k, v in proj.items() if not k.startswith("random")})
json.dump(res, open(f"{OUT}/results.json", "w"), ensure_ascii=False, default=float, indent=1)

html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><style>
body{{font-family:'WenQuanYi Zen Hei','Noto Sans CJK TC',sans-serif;font-size:12.5px;line-height:1.65;color:#1b1b1b;margin:0}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:20px 0 6px;border-bottom:2px solid #333;padding-bottom:3px}} h3{{font-size:13.5px;margin:12px 0 4px}}
table{{border-collapse:collapse;width:100%;margin:6px 0 8px;font-size:11.5px}} th,td{{border:1px solid #bbb;padding:4px 6px;text-align:left}} th{{background:#eee}} tr.sep td{{background:#f6f6f6;font-weight:bold}}
.box{{background:#fff7e0;border-left:5px solid #e69f00;padding:8px 12px;margin:8px 0}} .note{{font-size:11px;color:#555}} img{{width:100%}} .red{{color:#b03030;font-weight:bold}} li{{margin:2px 0}}
</style></head><body>
<h1>「AI炒股年化90%」是真的嗎?</h1><div class=note>2026-11-02 · 研究用,沒有改任何實盤規則 · 資料:Yahoo(還原價)、S&P500 歷史增刪紀錄重建的當時成分股</div>

<h2>一句話結論</h2>
<div class=box>我們不是找不到,是<b>90%在誠實條件下不存在</b>。我把常見的「輕鬆90%」做法逐一重現:它們確實能做出 40–140% 的數字,但每一個都靠下面其中一個機制:<b>用今天的贏家當股票池(倖存者/事後挑股)</b>、<b>只看最近的大牛市</b>、<b>3倍槓桿但不提回撤</b>、<b>不計成本</b>、<b>參數事後挑選</b>。換成「當年真正能買到的股票、扣成本、等一天成交」後,動能、AI選股、高頻全部掉到 <b>SPY 的水準(約 10–15%)或更低</b>。唯一能在誠實條件下做出 40%+ 的是 3 倍槓桿 QQQ(TQQQ),但它只是「牛市 Beta × 3」,代價是 <b>-82% ~ -95% 的回撤</b>。</div>
<img src="fig1_headline.png">

<h2>1. 這些「90%」都是怎麼來的</h2>
<table><tr><th>常見說法</th><th>製造高報酬的機制</th><th>我重現的結果(年化)</th></tr>
<tr><td>集中持有動能最強的5–10檔</td><td>股票池是「今天的」S&P500,裡面全是後來才漲到進指數的贏家</td><td>倖存者版 <span class=red>{pc(P(m['SURV|5|10'],FULL)['cagr'])}</span>(2023-26 <span class=red>{pc(P(m['SURV|5|10'],AI)['cagr'])}</span>)→ 當年成分股版 {pc(P(m['PIT|5|10'],FULL)['cagr'])}</td></tr>
<tr><td>「選到NVDA/TSLA…」少數幾檔</td><td>事後才知道哪幾檔會漲;過去贏家明年排名只在第 {ps.next_rank_pct.mean()*100:.0f} 百分位(=隨機)</td><td>NVDA 2023-26 年化 99%,但「去年最強10檔」隔年平均只贏大盤 {(ps.top10_next.mean()-ps.univ_next.mean())*100:.1f} 個百分點(20年裡贏 11 年)</td></tr>
<tr><td>2–3倍槓桿ETF(TQQQ/SOXL)</td><td>2010–2026 是史上最長牛市之一;槓桿放大 Beta,不放大技巧</td><td>TQQQ {pc(P(lev['TQQQ 真實'],FULL)['cagr'])}、2010-19 {pc(P(lev['TQQQ 真實'],OOS)['cagr'])},但回撤 {pc(P(lev['TQQQ 真實'],FULL)['mdd'])};2008年模擬 -95%</td></tr>
<tr><td>高頻/每天進出</td><td>回測用收盤價成交、零成本、零滑價</td><td>紙上 {pc(P(rv['紙上績效: 收盤進出、0成本'],FULL)['cagr'])} → 扣 10bps 變 <span class=red>{pc(P(rv['加 10bps 成本'],FULL)['cagr'])}</span>(年換手 {rv['加 10bps 成本']['turn_per_year']:.0f} 倍)</td></tr>
<tr><td>AI/機器學習選股</td><td>特徵看似很多,但預測力(IC)幾乎為零;偏差股票池讓它看起來有效</td><td>誠實版 {pc(ml['AI選股 前20名 (成分股, 10bps)']['full']['cagr'])} vs SPY {pc(ml['SPY']['full']['cagr'])};含偏差版 {pc(ml['AI選股 前20名 (含今日成分股偏差, 10bps)']['full']['cagr'])};IC t 值僅 {o3['ml_ic']['t']:.1f}</td></tr>
<tr><td>某組參數/某個策略「回測超強」</td><td>掃幾百組參數挑最好的(過度擬合),或只秀最好的一段</td><td>10檔事後贏家籃子:挑選期 {pc(sv['bh_is'])} → 之後 {pc(sv['bh_oos'])};掃 {sv['n']} 組均線,最佳組 IS {pc(sv['best']['is'])} → OOS {pc(sv['best']['oos'])}</td></tr></table>

<h2>2. 重現實驗(都用同一份資料、同一套成本)</h2>
<h3>A. 倖存者偏差:同一個動能策略,只換「股票池怎麼來」</h3>
{t_mom}
<img src="fig2_equity.png">
<p class=note>設定:每月底算12-1個月動能,買最強N檔等權,持有一個月;訊號日隔天再多等一天成交;每邊成本10bps。「當年成分股」= 用維基百科 S&P500 增刪紀錄回推每個月底的真實成分股(含後來被剔除、被購併的),有價格資料的覆蓋率 2005年約71%、2010年74%、2015年81%、2020年92% — 缺的是已下市的公司(通常是輸家),所以這個「誠實版」仍然<b>偏樂觀</b>。</p>
<p>重點:<b>同一套程式、同一個策略,倖存者版 {pc(P(m['SURV|5|10'],FULL)['cagr'])} vs 當年成分股版 {pc(P(m['PIT|5|10'],FULL)['cagr'])}</b>。連「隨機抽5檔」只要股票池是事後贏家,就有 {pc(np.mean([x[FULL]['cagr'] for x in m['HIND|5|10|rand']]))}。當年成分股版的動能只有 {pc(P(m['PIT|10|10'],FULL)['cagr'])}(前10檔),落在隨機抽取的 95 分位({pc(np.percentile([x[FULL]['cagr'] for x in m['PIT|10|10|rand']],95))})附近,且 2010-19 的 {pc(P(m['PIT|10|10'],OOS)['cagr'])} 低於 SPY 的 {pc(spyr_[OOS]['cagr'])}。</p>

<h3>B. 滾動窗口:「輕鬆90%」其實是挑到的那一年</h3>
{t_roll}
<p>在當年成分股裡,<b>沒有任何一檔股票</b>能連續 5 年年化 ≥90%(2005–2021 每年起算,共 {int(fy.n.sum())} 檔次;最接近的是 NVDA 2019→2024 約 87%)。單年 ≥90% 的股票每年約 {ps.share_ge90.mean()*100:.1f}%,而且「去年的前10名」今年表現是隨機的。</p>

<h3>C. 槓桿ETF:能做到,但那是Beta不是技巧</h3>
{t_lev}
<img src="fig3_leverage_dd.png">
<p>隱藏風險:單日最糟 {pc(ex['TQQQ 真實']['worst_day'])};波動衰減(震盪盤每天重置會侵蝕);加200日線進出<b>沒有明顯改善</b>(2010-19 {pc(P(lev['QQQ 3x + 200日線進出'],OOS)['cagr'])},回撤仍 {pc(P(lev['QQQ 3x + 200日線進出'],OOS)['mdd'])}),因為假突破來回被洗。我們前一個研究也已確認個股2倍ETF的成本年化約17%。</p>

<h3>D. 成本:高頻策略的利潤全被吃掉</h3>
{t_rev}
<h3>E. 「AI」選股:機器學習有沒有真的找到東西</h3>
{t_ml}
<p>做法:15個特徵(1/3/6/12-1個月動能、5日反轉、60日波動、距52週高、成交值,與其橫斷面排名)→ 梯度提升樹預測下月相對報酬;從2011年起每年1月只用「標籤已經結束」的資料重訓。預測力(Spearman IC)平均 {o3['ml_ic']['mean']:.3f},t = {o3['ml_ic']['t']:.1f}(門檻 2.5)→ <b>沒有可靠預測力</b>。2010-19 看起來比 SPY 好一點,2020-26 反而輸。把今天成分股混進候選池(偏差)就立刻跳到 {pc(ml['AI選股 前20名 (含今日成分股偏差, 10bps)']['full']['cagr'])} — 看起來很強的「AI」多半是這種偏差。</p>

<h3>F. 少數股票 + 掃參數 = 過度擬合</h3>
<p>挑 2015–2020 漲最多的 10 檔({', '.join(o3['basket'])}),那段時間買入持有年化 <b>{pc(sv['bh_is'])}</b>;之後 2021–26 只剩 <b>{pc(sv['bh_oos'])}</b>(SPY {pc(sv['spy_oos'])})。再對這 10 檔掃 {sv['n']} 組均線交叉:選樣期最佳 {pc(sv['best']['is'])}(參數 {int(sv['best']['fast'])}/{int(sv['best']['slow'])}),之後 <b>{pc(sv['best']['oos'])}</b>;全部組合中只有 {sv['share_oos_gt_spy']*100:.1f}% 在之後贏過 SPY。選樣期與之後績效的相關係數 {sv['corr']:.2f}。</p>

<h3>G. 我們自己的股票池也有同樣問題</h3>
<p>終端的 universe 是 2026-10-31 用「現在」的 Beta≥1.3、波動≥35%、成交值≥1億篩出來的 {170} 檔,屬事後篩選。在這個池子裡<b>隨機抽10檔</b>(10bps)年化 {pc(np.mean([x[FULL]['cagr'] for x in o4['random|10']]))}(2010-19 {pc(np.mean([x[OOS]['cagr'] for x in o4['random|10']]))}),比當年成分股隨機抽的 {pc(np.mean([x[FULL]['cagr'] for x in m['PIT|10|10|rand']]))} 高出約 7 個百分點 — 這是股票池給的免費紅利。池內動能前10檔 {pc(P({'periods':o4['momentum|10']},FULL)['cagr'])}(2010-19 {pc(P({'periods':o4['momentum|10']},OOS)['cagr'])}),仍落在隨機抽取的 95 分位({pc(np.percentile([x[OOS]['cagr'] for x in o4['random|10']],95))})以內。先前研究裡常見的 60–70% 年化與「隨機進場也有 +59%」是逐筆交易的年化,和這裡的帳戶複利不同,不能直接相比;但方向一致:<b>要看的是相對隨機的超額,不是絕對數字</b>。帳戶層級的誠實數字(10檔、1%風險)是全期約 12%、2010-19 約 -3%,低於 SPY。</p>

<h2>3. 哪些活下來、各自藏了什麼風險</h2>
<table><tr><th>做法</th><th>誠實條件下年化</th><th>是否勝過SPY/隨機</th><th>隱藏風險</th></tr>
<tr><td>3倍槓桿 QQQ(TQQQ)</td><td>{pc(P(lev['TQQQ 真實'],FULL)['cagr'])}(2010-19 {pc(P(lev['TQQQ 真實'],OOS)['cagr'])})</td><td>勝過SPY(單一牛市)</td><td>回撤 {pc(P(lev['TQQQ 真實'],FULL)['mdd'])};2008同款 -95%;水下 {ex['TQQQ 真實']['years_underwater']:.0f}–{ex['QQQ 3x 買入持有(模擬)']['years_underwater']:.0f} 年;樣本只有一個牛市</td></tr>
<tr><td>當年成分股動能前10</td><td>{pc(P(m['PIT|10|10'],FULL)['cagr'])}</td><td>≈SPY、落在隨機區間內</td><td>回撤 {pc(P(m['PIT|10|10'],FULL)['mdd'])}(SPY {pc(spyr_[FULL]['mdd'])});偏樂觀(缺下市股)</td></tr>
<tr><td>AI機器學習選股</td><td>{pc(ml['AI選股 前20名 (成分股, 10bps)']['full']['cagr'])}</td><td>≈SPY,IC不顯著</td><td>{pc(ml['AI選股 前20名 (成分股, 10bps)']['full']['mdd'])} 回撤;2020-26 反而輸 SPY</td></tr>
<tr><td>高頻反轉</td><td>{pc(P(rv['加 10bps 成本'],FULL)['cagr'])}</td><td>輸</td><td>成本吃光</td></tr>
<tr><td>參數掃描/少數贏家</td><td>{pc(sv['best']['oos'])}</td><td>輸</td><td>過度擬合</td></tr></table>
<div class=box><b>所以「年化90%」的真實面貌:</b>① 不是策略,是挑了對的股票池或時間窗;② 想複製必須承受 -80%~-95% 的回撤(槓桿)或靠事後才知道的贏家;③ 一旦回到「當年買得到的股票 + 成本 + 延遲一天」,可持續的水準大約是<b>未槓桿 10–15%、3倍Beta約 40%(含 -80% 回撤)</b>。</div>

<h2>4. 方法與限制</h2>
<ul>
<li>資料:Yahoo 日線(還原股息/分割);當時成分股由維基百科 S&amp;P500 增刪紀錄回推(2005起);已下市股票 Yahoo 沒有資料,所以誠實版仍略偏樂觀(覆蓋率 71%→92%)。</li>
<li>訊號日收盤決策、再多等一天才成交;成本每邊 10bps(另測 0/25bps);被收購下市視為以最後價格轉現金。</li>
<li>期間:2005–2026-10(2026 為部分年)。槓桿ETF模擬= 日報酬×倍數 − 費用0.95% − 融資成本(逐年T-bill + 0.5%);TQQQ 模擬與真實年化 {pc(o2['valid_TQQQ']['synthetic'])} vs {pc(o2['valid_TQQQ']['real'])}。SOXL 以 SMH 模擬會高估(43% vs 真實 32%),所以只引用真實 SOXL 的數字。</li>
<li>沒有測選擇權(沒有歷史選擇權價格);選擇權的高報酬同樣來自槓桿 + 方向押注,風險只會更高,這裡不下結論。</li>
<li>已禁用且未重測:分批進場(scaling-in);未改任何實盤規則。程式:<code>reality/</code>。</li>
</ul>
</body></html>"""
open(f"{OUT}/report.html", "w").write(html)
