# 最優策略組合研究：進度（2026-11-04 11:31 UTC 應 Neo 要求暫停）

Neo 要求：「結合你所有數據同知識，幫我無限優化」。做法係大規模搜尋 x80 組合嘅各種設定，但用鎖起嘅最終考試期（2025-01 至今）
同預先寫低嘅揀選規則防止過度優化（`PREREG.md`）。純研究，冇改實盤設定。

## 已完成

* 候選池 30 個策略（`src/backtest/optimize/pool.py`）：x80 第二輪 16 個 + 3 個新策略 + 11 個之前幾輪嘅變體。
  新策略喺 2025 年之前（考試期之前）嘅每週超額回報 Sharpe：RSI(2) 均值回歸 1.29（2007 起；同 x80 組合相關 0.51）、
  G10 外匯套息 0.22（純多空回報，2004 起）、月尾月頭 0.25。RSI(2) 用嘅係 Yahoo 仲有數據嘅歷史成分股，已下市嘅股票缺失，會高估撈底策略。
* 組合引擎（`engine.py`）：8 種權重法、3 種協方差、揀策略規則、單一策略同加密貨幣風險上限、再平衡同槓桿容差帶；
  同 x80 原版逐週對過（組合回報差 < 1e-10）。11 個測試通過（`tests/test_optimize.py`）。
* 搜尋、過度優化檢查、最終考試嘅程式已寫好（`search.py`、`overfit.py`、`final.py`），但未跑完。

## 未完成

* 10,026 個組合嘅搜尋跑到約 4,000 個時停咗；結果喺程式最尾先寫出，所以今次冇存到搜尋結果。
* 過度優化檢查（walk-forward 重新優化、CSCV 過擬合機率、deflated Sharpe）、最終考試、報告都未做。

## 初步數字（只係 x80 原版同「只改一樣」嘅 25 個版本，未做過度優化檢查，唔好當結論）

槓桿到 80% 目標波動、扣晒成本，Sharpe（超額回報）：

| 改動 | 2010-17 | 2018-24 | 2010-24 |
|---|---|---|---|
| x80 原版（等風險貢獻、104 週、16 個策略） | 1.34 | 1.16 | 1.22 |
| 平均分權重（ew） | 1.61 | 1.27 | 1.44 |
| 單一策略上限 15% | 1.44 | 1.19 | 1.32 |
| 每 4 週先再平衡 | 1.34 | 1.22 | 1.26 |
| 槓桿容差帶 10% | 1.36 | 1.20 | 1.25 |
| 加 3 個新策略 | 1.24 | 1.19 | 1.21 |
| 全部 30 個策略 | 0.35 | 0.84 | 0.59 |
| 階層風險平價 HRP | 0.93 | 0.66 | 0.78 |

平均分權重會畀高波動嘅加密貨幣策略多好多風險，佢好過原版可能只係因為 2010-24 加密貨幣表現好，要過咗過度優化檢查同最終考試先知。

## 點樣繼續

```bash
R=/home/user/research_data; O=$R/opt
python -I scripts/optimize/fx_dl.py $O/raw
PYTHONPATH=. python -m src.backtest.optimize.newsleeves $R/raw $R/x80/raw $O/raw $O/new_sleeves.parquet
PYTHONPATH=. python -m src.backtest.optimize.pool $R/x80/out $R/x80/raw $O/new_sleeves.parquet $O
PYTHONPATH=. python -m src.backtest.optimize.search $O $R/x80/out/cash_weekly.parquet $O/search 10000   # 4 核約 27 分鐘
PYTHONPATH=. python -m src.backtest.optimize.overfit $O/search $R/x80/out/cash_weekly.parquet $O/overfit
PYTHONPATH=. python -m src.backtest.optimize.final $O $R/x80/out/cash_weekly.parquet $O/search $O/overfit $O/final
```

需要 x80 同 levscore 研究嘅數據先（`docs/X80.md`、`docs/LEVSCORE.md` 嘅重現步驟）；研究數據唔喺 repo 入面。
