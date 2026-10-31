"""Traditional Chinese labels for the research terminal (web dashboard + daily report).

Internal keys stay English (they are used in logic, the database and tests); only the
DISPLAY text is translated. ``tr(x)`` returns the Chinese label or ``x`` unchanged.
"""

from __future__ import annotations

LABELS: dict[str, str] = {
    # market regime
    "Strong Bullish": "強勢看多", "Bullish": "看多", "Neutral": "中性", "Bearish": "看空", "Strong Bearish": "強勢看空",
    "Index trend": "指數趨勢", "Index momentum": "指數動量", "Breadth (universe)": "市場寬度（股票池）",
    "Volatility (VIX)": "波動率（VIX）", "Risk appetite": "風險偏好", "Rates": "利率",
    # sector rotation
    "Leading": "領先", "Weakening": "轉弱", "Lagging": "落後", "Improving": "改善",
    "Technology": "科技", "Communication Services": "通訊服務", "Consumer Discretionary": "非必需消費",
    "Consumer Staples": "必需消費", "Energy": "能源", "Financials": "金融", "Health Care": "醫療保健",
    "Industrials": "工業", "Materials": "原材料", "Real Estate": "房地產", "Utilities": "公用事業",
    "Semiconductors": "半導體", "Biotech": "生物科技", "Regional Banks": "地區銀行", "Aerospace & Defense": "航太與國防",
    # index table
    "bullish": "多頭排列", "bearish": "空頭排列", "mixed": "混合",
    # swing trend (EMA9 / EMA21)
    "Strong short-term uptrend": "強勢短線上升", "Short-term uptrend": "短線上升", "Mixed": "混合",
    "Deteriorating": "轉弱", "Short-term downtrend": "短線下跌", "Strong short-term downtrend": "強勢短線下跌",
    # swing structure / condition
    "HH/HL": "高點低點抬高", "LH/LL": "高點低點下移",
    "Strong": "強", "Weak": "弱",
    # structure states
    "Breakout with volume confirmation": "放量突破", "Breakout on normal volume": "正常量突破",
    "Breakout without volume confirmation": "無量突破", "Short-term breakdown with heavy volume": "放量短線破位",
    "Short-term breakdown": "短線破位", "Price losing EMA21": "跌穿 EMA21", "Price reclaiming EMA21": "重返 EMA21",
    "EMA9 crossed below EMA21": "EMA9 下穿 EMA21", "EMA9 crossed above EMA21": "EMA9 上穿 EMA21",
    "Pullback to EMA9": "回踩 EMA9", "Pullback to EMA21": "回踩 EMA21", "Pullback within swing uptrend": "上升趨勢中回調",
    "Price losing EMA9": "跌穿 EMA9", "Price reclaiming EMA9": "重返 EMA9", "Near breakout": "接近突破",
    "Momentum deterioration": "動量轉弱", "EMA9 / EMA21 compression": "EMA9 / EMA21 收窄", "Volatility squeeze": "波動收窄",
    "Range / consolidation": "區間整理", "Price > EMA9 > EMA21": "價格 > EMA9 > EMA21",
    "Price < EMA9 < EMA21": "價格 < EMA9 < EMA21", "No distinct structure": "無明顯結構",
    # volatility / relative strength labels
    "Extreme": "極高", "High": "高", "Elevated": "偏高", "Normal": "正常", "Low": "低",
    "Strongly outperforming": "大幅跑贏", "Outperforming": "跑贏", "Underperforming": "跑輸",
    "Strongly underperforming": "大幅跑輸",
    # candidates
    "long": "做多", "short": "做空", "Watch": "觀察",
    "trend": "趨勢", "momentum": "動量", "relative_strength": "相對強弱", "volume": "成交量",
    "setup_quality": "形態質素", "reward_risk": "風險報酬", "liquidity": "流動性", "regime_alignment": "市場配合度",
    "earnings_in_window": "持倉期內有財報", "extended": "過度延伸", "overbought_extreme": "RSI 極端",
    "dilution_filing": "稀釋相關文件", "financial_distress": "財務壓力", "poor_reward_risk": "風險報酬比偏低",
    "wide_stop": "止損過寬", "data_quality": "數據品質", "options_event_premium": "期權事件溢價",
    "ALL_LONG": "全部做多形態", "ALL_SHORT": "全部做空形態",
    # catalysts / statuses
    "Positive": "正面", "Negative": "負面", "Uncertain": "不確定",
    "Confirmed": "已確認", "Estimated": "預估", "Speculative": "推測",
    "before open": "開市前", "after close": "收市後", "during session": "交易時段",
    "ok": "正常", "warning": "警告", "excluded": "已排除",
    "success": "成功", "success_with_warnings": "成功（有警告）", "failed": "失敗", "running": "運行中",
    "risk": "風險", "strength": "優勢",
    # holidays
    "New Year's Day": "元旦", "New Year's Day (observed)": "元旦（補假）", "Martin Luther King Jr. Day": "馬丁路德金紀念日",
    "Washington's Birthday": "總統日", "Good Friday": "耶穌受難日", "Memorial Day": "陣亡將士紀念日",
    "Juneteenth": "六月節", "Independence Day": "獨立日", "Labor Day": "勞動節", "Thanksgiving Day": "感恩節",
    "Christmas Day": "聖誕節",
    "Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三", "Thursday": "星期四", "Friday": "星期五",
    "Saturday": "星期六", "Sunday": "星期日",
}


def tr(value: object) -> str:
    """Chinese display label for an English key (unchanged when there is no translation)."""
    text = "" if value is None else str(value)
    return LABELS.get(text, text)


METHODOLOGY: dict[str, str] = {
    "regime": """市場狀態由六個項目加權而成。每項按固定公式評分（−2 至 +2），綜合分數 = Σ 權重 × 分數 / 2，範圍 −100 至 +100。

分級：≥ +45 強勢看多｜≥ +15 看多｜> −15 中性｜> −45 看空｜其餘為強勢看空。
「強勢」級別要求 SPY 位於 EMA50 相應一方，否則降一級。

1. 指數趨勢（權重 25）：SPY、QQQ、IWM、DIA 各自計 0.5 ×［價格相對 EMA21、相對 EMA50、EMA50 相對 EMA200、EMA21 三日斜率］的正負號總和；
   SPY 40%、QQQ 30%、IWM 15%、DIA 15%。
2. 指數動量（15）：SPY 和 QQQ 的 20 日回報 ÷ 其正常波幅（60 日日波動 × √20），上限 ±2，取平均。
3. 市場寬度（20）：股票池內高於 EMA50 的比例、高於 EMA200 的比例、創 20 日新高減新低的比例（以股票池估算，並非全市場）。
4. 波動率（20）：VIX 水平（<15 +2、<18 +1、<22 0、<28 −1、其餘 −2）；VIX/VIX3M > 1（期限結構倒掛）−1，< 0.90 +0.5；VIX 5 日急升 > 25% −0.5。
5. 風險偏好（15）：0.4 × 高收益債減國債（HYG−IEF）20 日回報 ÷ 1% + 0.4 × 進攻型減防守型板塊 20 日回報 ÷ 2% + 0.2 × IWM 減 SPY 20 日回報 ÷ 2%。
6. 利率（5）：10 年期國債收益率 20 日變化：> +30 基點 −2、> +15 −1、< −15 +1，其餘 0。

缺少的項目不計入，其餘項目重新加權。板塊輪動：RS = ETF / SPY；RS 趨勢 = RS / RS 的 50 日均線 − 1；RS 動量 = RS 趨勢的 10 日變化；
按兩者正負分為領先、轉弱、落後、改善四個象限。以上全部是描述，不是預測。""",
    "setups": """形態是多項證據的組合（趨勢背景＋位置＋成交量＋動量），從不只看單一指標交叉。

做多形態
• 突破：收市價高於前 20 日最高價，成交量 ≥ 1.3 倍 MAVOL20，收市位於當日波幅上方（CLV ≥ 0.6），價格高於 EMA50。
• 上升趨勢回踩 EMA9/21：EMA21 > EMA50、EMA50 向上、價格高於 EMA50；近 3 日低位觸及 EMA21 或 EMA9 0.5 ATR 內；
  距 10 日高位至少 1 ATR；RSI14 在 40 至 62；回調期間成交量偏低。
• 高位波動收窄：布林帶寬處於過去 126 日最低 15%，價格 > EMA21 > EMA50，距 20 日高位 1.5 ATR 內；只在突破 20 日高位時才進場（條件進場）。
• 重返 EMA21：在 EMA21 下方至少 3 日後，今日或昨日收復 EMA21，成交量 ≥ MAVOL20，且價格高於 EMA50 或 EMA50 向上。
• 強勢股貼近高位：價格 > EMA9 > EMA21 > EMA50，20 日及 60 日跑贏 SPY，距 20 日高位 1 ATR 內，距 EMA21 不超過 2.5 ATR。

做空形態（鏡像）
• 跌破：收市價低於前 20 日最低價，放量，收市位於當日波幅下方，價格低於 EMA50。
• 反彈受阻於下行 EMA21：EMA21 < EMA50、EMA50 向下；近 3 日高位觸及 EMA21 附近後回落；RSI14 在 38 至 60；反彈量縮。
• 跌穿 EMA21：在 EMA21 上方至少 3 日後跌穿，成交量 ≥ MAVOL20，且價格低於 EMA50 或 EMA50 向下。

每個形態另有 0-100 的質素分，按成交量、收窄程度、結構等列明的條件加分。""",
    "plan": """風險計劃（今日報告和歷史回測用同一套規則）：

• 參考進場：今日收市價（回測用翌日開市價，包含跳空）。波動收窄形態為條件進場：升穿 20 日高位 + 0.1 ATR。
• 結構止損：突破＝當日低位與前 20 日高位較低者 − 0.25 ATR；回踩＝近 5 日低位與 EMA21 較低者 − 0.25 ATR；
  收窄＝近 5 日低位 − 0.25 ATR；重返＝近 3 日低位 − 0.25 ATR；強勢＝EMA21 − 0.5 ATR 與近 5 日低位 − 0.25 ATR 較高者。
• 止損距離限制在 1 至 3 個 ATR；結構止損超過上限時會標示「止損過寬」。
• 目標一：進場價上方至少 0.75 ATR 的最近 20 日／60 日／252 日高位；如果已在高位，則為進場價 + 2 ATR。
• 目標二：目標一 + 1 ATR 與 2 倍風險兩者較高者。
• 風險報酬比 =（目標一 − 進場）÷（進場 − 止損）。時間止損：7 個交易日。做空完全鏡像。""",
    "scoring": """綜合分數 = Σ 權重 × 分項分數 / 100 − 扣分（0 至 100）。分項分數都按交易方向計算（做空時趨勢向下得高分）。

• 趨勢（20）：價格 > EMA21、EMA21 > EMA50、EMA50 > EMA200、EMA21 向上、EMA50 向上、高點低點抬高。
• 動量（15）：RSI14 在 50 至 70 最好，過熱扣分；MACD 柱 > 0、柱狀向上、10 日變動率 > 0。
• 相對強弱（15）：相對 SPY 的強弱在股票池的百分位（70%）＋相對板塊 ETF 的百分位（30%）。
• 成交量（10）：20 日上漲／下跌成交量比（40%）、OBV 斜率（30%）、按形態要求放量或縮量（30%）。
• 形態質素（15）、風險報酬（10）、流動性（5）、市場配合度（10：市場狀態 60% ＋ 所屬板塊象限 40%）。

扣分：持倉期內有財報、過度延伸（距 EMA21 超過 3 ATR）、RSI6 極端、近期股權稀釋文件（只對做多）、財務壓力（只對做多）、
風險報酬比低於 1.5、止損過寬、數據品質警告、期權隱含波動率遠高於實際波動率。

級別：A ≥ 75、B ≥ 62、C ≥ 50，其餘為觀察。級別代表研究優先次序，不是交易建議。

高 Beta／高波動篩選（每日）：120 日 Beta ≥ 1.3、60 日年化波動率 ≥ 35%、市值 ≥ 20 億美元（槓桿 ETF 豁免）、
20 日平均成交額 ≥ 1 億美元；任何一項不符合即不顯示（原因見「數據」頁）。""",
    "backtest": """每個形態的計劃都在股票池的歷史數據上模擬：

1. 收市出現信號；所有指標只用當日或之前的數據（沒有偷看未來）。
2. 翌日開市進場，並扣除滑點；條件進場只在翌日觸及觸發價時才成交；開市已越過止損則不進場。
3. 其後每日：觸及止損即止損出場；觸及目標一即獲利出場；同一根 K 線同時觸及兩者時當作先止損（保守）。
4. 否則在第 7 個交易日收市出場。

結果以 R 倍數（盈虧 ÷ 初始風險）和百分比顯示，按形態及信號當日的市場狀態分組，並與「所有股票日」的基準比較。

已知偏差：倖存者偏差（使用今日的股票池）、樣本內（規則和數據重疊）、相關股票的信號重疊、未計佣金和借券成本、只用日線。
請把這些統計當作規則的合理性檢查，而不是未來回報的預期。""",
}
