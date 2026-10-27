# Backtest results

Data: daily bars 2022-01-13 .. 2026-10-26 (1200 sessions). IS = before 2024-07-01, OOS = after. Costs 5 bp/side, cash 4%, margin 10% (assumptions). Past results of a 4.8-year window that contains a semiconductor bull market; not a forecast.


## 1. Candidates vs buy & hold

| Strategy | CAGR | Vol | Sharpe | MaxDD | CAGR IS | MaxDD IS | CAGR OOS | MaxDD OOS |
|---|---|---|---|---|---|---|---|---|
| Buy & hold SPY | +13% | +17% | 0.54 | -24% | +9% | -24% | +17% | -20% |
| Buy & hold SMH (semis, 1x) | +31% | +37% | 0.81 | -44% | +27% | -44% | +37% | -38% |
| Buy & hold SOXL (3x semis) | +9% | +116% | 0.62 | -89% | +0% | -89% | +23% | -89% |
| SOXL 127% of equity, margin (current style) | -14% | +147% | 0.61 | -96% | -20% | -95% | -5% | -95% |
| SOXL when SMH > SMA200, else cash | +48% | +92% | 0.85 | -74% | +69% | -47% | +33% | -74% |
| SOXL vol-target 20% | +21% | +22% | 0.80 | -30% | +23% | -21% | +19% | -30% |
| SOXL vol-target 30% | +28% | +32% | 0.80 | -43% | +31% | -31% | +25% | -43% |
| SOXL vol-target 40% | +34% | +43% | 0.80 | -54% | +39% | -40% | +30% | -54% |
| SOXL vol-target 20% + SMH>SMA200 | +23% | +19% | 0.98 | -20% | +25% | -13% | +22% | -20% |
| SOXL vol-target 30% + SMH>SMA200 | +32% | +28% | 0.98 | -30% | +35% | -20% | +30% | -30% |
| SOXL vol-target 40% + SMH>SMA200 | +40% | +37% | 0.98 | -39% | +45% | -26% | +37% | -39% |
| SOXL vol-target 20% + SMH>SMA200 + SOXL>SMA50 | +20% | +15% | 1.00 | -20% | +23% | -7% | +17% | -20% |
| SOXL vol-target 30% + SMH>SMA200 + SOXL>SMA50 | +27% | +22% | 1.00 | -30% | +32% | -10% | +23% | -30% |
| SOXL vol-target 40% + SMH>SMA200 + SOXL>SMA50 | +34% | +30% | 1.00 | -39% | +42% | -13% | +29% | -39% |
| A: vol-target 30% + SMH>SMA200 + circuit breaker | +33% | +26% | 1.06 | -21% | +29% | -18% | +38% | -21% |
| A2: same + SOXL>SMA50 + circuit breaker | +30% | +22% | 1.12 | -23% | +32% | -10% | +29% | -23% |

## 2. Robustness: length of the SMH trend gate (vol-target 30%)

| SMH SMA length | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| 100 | +24% | 0.80 | -39% |
| 125 | +24% | 0.78 | -41% |
| 150 | +24% | 0.78 | -46% |
| 175 | +29% | 0.90 | -33% |
| 200 | +32% | 0.98 | -30% |
| 225 | +31% | 0.95 | -34% |
| 250 | +31% | 0.95 | -33% |

## 3. Calendar years (return)

| date | SOXL buy&hold | SMH buy&hold | A (30% VT + SMH>SMA200 + breaker) |
|---|---|---|---|
| 2022 | -83% | -31% | -5% |
| 2023 | +171% | +63% | +46% |
| 2024 | +7% | +49% | +18% |
| 2025 | +79% | +55% | +63% |
| 2026 | +71% | +39% | +43% |

## 4. Block-bootstrap (20-day blocks, 5000 one-year paths, resampled from this same window)

| Strategy | median 1y | 5th pct 1y | P(1y loss) | median maxDD | 95th pct worst DD | P(DD worse than -30%) |
|---|---|---|---|---|---|---|
| A | +35% | -8% | 10% | -15% | -27% | 2% |
| SOXL buy&hold | +30% | -79% | 40% | -66% | -89% | 100% |
| SOXL 127% margin | +3% | -90% | 49% | -78% | -95% | 100% |

## 5. Short-term SOXL setups (one position at a time, full position, per-trade return)

| Setup | N | Win% | Avg | Median | Avg win | Avg loss | Worst | Avg IS (N) | Avg OOS (N) |
|---|---|---|---|---|---|---|---|---|---|
| RSI2<10, SMH>SMA200 (exit > prior high / 5d / -10%) | 42 | 64% | +1.3% | +2.3% | +6.8% | -8.6% | -20.1% | +1.8% (18) | +0.9% (24) |
| RSI2<20, SMH>SMA200 (same exit) | 86 | 63% | +2.5% | +3.5% | +9.6% | -9.5% | -23.5% | +3.8% (37) | +1.5% (49) |
| 3 down closes, SMH>SMA200 (same exit) | 42 | 69% | +2.6% | +5.2% | +8.1% | -9.5% | -20.1% | +3.3% (21) | +1.9% (21) |
| 20-day-high breakout, SMH>SMA200 (exit < EMA21 / 20d / -12%) | 33 | 33% | +3.8% | -8.0% | +33.7% | -11.2% | -23.7% | +6.9% (13) | +1.7% (20) |
| Random entries, same exit as RSI2 setups (control) | 57 | 53% | +0.3% | +2.1% | +8.9% | -9.3% | -23.7% | +1.4% (26) | -0.6% (31) |

## 6. Signal state at the last bar

- date: 2026-10-26
- SOXL close: 90.5
- SOXL 20d vol: 120%
- SMH>SMA200: True
- SMH>SMA100: False
- SMH>SMA50: False
- SOXL>SMA50: False
- weight A (share of equity in SOXL): 25%
