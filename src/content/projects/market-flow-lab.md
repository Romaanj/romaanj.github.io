---
title: Market Flow Lab
summary: >-
  Reading the market's tape the way I read a GPU profiler trace: regime
  detection on stock and crypto time series, order-flow and momentum signals,
  and backtests honest enough to hurt. If a signal can't survive transaction
  costs and a shuffled-label null test, it doesn't count as a signal.
status: idea
date: 2026-07-04
tags: [markets, data, timeseries]
---

Still on the whiteboard. The rough plan:

- **Regime detection** — segment price/volume series into trending, mean-reverting, and chop states first, instead of asking one signal to work everywhere.
- **Flow signals** — order-flow imbalance and momentum with explicit decay. Nothing that needs a narrative to justify its existence.
- **Honest backtests** — walk-forward splits, fees and slippage on by default, and a null-model baseline so luck has to announce itself before it gets called alpha.

Mostly an excuse to point time-series tooling at something that fights back.
