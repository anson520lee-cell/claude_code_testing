# Edge hunt（2026-11-03）：最後一輪找能改善策略的方法

Neo 的要求：「基於現在的資訊，找出能改善現在交易策略的方法」，可以用任何指標、自創指標、籌碼、期權，
做多做空都要測，把網上不同波段策略的長處結合，並逐個解釋為什麼失效、能否用現有資料優化。
完整中文報告（含圖）：project files `edge-hunt/最後一輪策略研究報告.pdf`。

## 結論

在誠實條件下（當年 S&P 500 成分股、扣成本、對比同日隨機入場、樣本外、Bonferroni 校正，再用從未碰過的
S&P 400/600 股票池重做），**沒有找到能讓策略穩定跑贏 SPY 的方法**。只有兩個假設通過獨立驗證：

| 假設 | S&P 500（發現） | S&P 400/600（獨立驗證，預先登記） |
|---|---|---|
| H1 隨機做多 × FINRA 5 日沽空量比在當日最低三分一 | 2010-17 +0.041R (t 3.3)，2018-26 +0.027R (t 2.3) | +0.026R，t 3.66（前後段 t 3.58 / 2.19）**通過** |
| A1 所有做空策略 × 特異波動最高三分一 | 同方向，個別策略 t 2-3 | +0.059R，t 3.79 **通過**，但做空仍每筆 −0.18R |

另外：突破與 squeeze 兩個現行 setup 在兩個股票池都輸給隨機入場（S&P 400/600 t −3.5 / −3.2）。

賬戶模擬（10 格、1% 風險、beta ≥ 1.3 且 hv60 ≥ 35% 的股票，2010-26，年化 / 最大回撤 / Sharpe）：

| 版本 | S&P 500 當年成分股 | S&P 400/600 |
|---|---|---|
| SPY | 13.9% / −34% / 1.01 | 同左 |
| 現行三個 setup | 9.5% / −59% / 0.46 | −7.3% / −94% / 0.01 |
| 現行 + 只做沽空量比低 | 7.9% / −22% / 0.48 | 20.7% / −33% / 0.79 |
| 去掉突破 | 13.3% / −31% / 0.61 | 4.2% / −60% / 0.31 |
| 去掉突破 + 只做沽空量比低 | 7.4% / −19% / 0.45 | 12.3% / −27% / 0.65 |

## 改了什麼

* `scanner.finra_low_short_volume`（config/terminal.yaml）：長倉候選的 5 日沽空量比在當天 S&P 500
  （`config/finra_reference.txt`）最低三分一時 +3 分（= 150 分/R × 0.020R，四段時間中最小的那段）。
  `penalty_above` 預設 0；設成例如 30 = 過濾（不在最低三分一的長倉扣分）。
* `src/positioning/positioning.py`：`FinraData.ratio5` / `short_volume_rank`，用終端本來就下載的 FINRA CNMS 每日檔。
* 沒有改 setup 清單、止損止盈、做空設定。是否去掉突破、是否開過濾，由 Neo 決定。

## 測了什麼

* 30 個網上做多策略（swing42 的 29 個 + 年底稅損）、10 個做空策略（`zoo.py`）、現行 3 多 3 空、隨機多空基準。
* 新資料：FINRA 每日沽空量與場外佔比、SEC XBRL 財報（按申報日）、8-K 財報日、Yahoo EPS 驚喜與分析師評級、
  VIX / VVIX / SKEW / VIX9D / 股票認沽認購比、自創指標（暴跌中大市佔比、隔夜佔比、跌速、特異波動、MAX、
  12-1 動量、季節性、連續暴跌）、29 個策略的共振與加權共振、故事組合。
* 3,942 個「策略 × 條件」：Tier 1（現行 setup + 隨機做多）334 個，0 個通過；Tier 2 3,608 個，3 個通過但都不可信。
* 預先登記 12 個假設（原文：`docs/EDGE_HUNT_PREREG.md`）在 871 隻 S&P 400/600 股票重做：2 個通過。

## 重現

```bash
# S&P 500 當年成分股
python -m src.backtest.edge_hunt.build --prices raw/prices.parquet --hist raw/sp500_hist.csv --out signals_v2.parquet
python -m src.backtest.edge_hunt.conf_build --prices raw/prices.parquet --hist raw/sp500_hist.csv --signals signals_v2.parquet --out confluence.parquet   # 共振權重（樣本內 2010-17）
python -m src.backtest.edge_hunt.grid --prices raw/prices.parquet --hist raw/sp500_hist.csv --signals signals_v2.parquet --out v2
python -m src.backtest.edge_hunt.grid_sources --raw raw --dir v2
python -m src.backtest.edge_hunt.run2 --dir v2 --signals signals_v2.parquet --out v2/out --prices raw/prices.parquet
python -m src.backtest.edge_hunt.why --out v2/out --rep rep/out
# 獨立驗證：S&P 400/600，成交額 >= $5M
python -m src.backtest.edge_hunt.build --prices rep/raw/prices.parquet --hist ALL:5e6 --out rep/signals.parquet
python -m src.backtest.edge_hunt.grid --prices rep/raw/prices.parquet --hist ALL:5e6 --signals rep/signals.parquet --out rep
python -m src.backtest.edge_hunt.replicate --rep rep --raw rep/raw --out rep/out
# 賬戶模擬與圖
python -m src.backtest.edge_hunt.account_run --signals signals_v2.parquet --prices raw/prices.parquet --out v2/acct --flags v2/flow_flags.parquet
python -m src.backtest.edge_hunt.figures --v2 v2 --rep rep --out OUT
```

資料下載腳本：`scripts/edge_hunt/`（SEC 請求用通用 User-Agent）；結果表：`reports/edge_hunt/`。
