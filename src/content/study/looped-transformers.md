---
title: 'Looped transformers: depth as a runtime knob'
title_ko: 'Looped Transformer: 런타임 노브가 된 깊이'
track: models
summary: 'An interactive explorer of looped / recurrent-depth transformers — the prelude–core–coda sandwich, how the loop is fed (Huginn’s concat-adapter, Parcae’s decay-SSM injection, LOTUS’s additive latents), why naive loops explode, random-depth training, test-time depth scaling with early exit, KV caching along the loop axis, and the params-vs-FLOPs-vs-bytes accounting.'
summary_ko: 'Looped / recurrent-depth transformer 인터랙티브 해부 — prelude–core–coda 샌드위치, loop에 입력을 먹이는 법 (Huginn의 concat-adapter, Parcae의 decay-SSM injection, LOTUS의 additive latent), naive loop가 폭발하는 이유, random-depth 학습, early exit을 포함한 test-time depth scaling, loop 축의 KV caching, 그리고 params-FLOPs-bytes 장부 정리.'
date: 2026-07-07
order: 4
interactive: /study/models/looped-transformers/
tags: [architecture, reasoning, looped-lm]
---

A looped transformer stores a few layers once and applies them many times: prelude →
weight-tied core × r → coda. Parameters stop tracking depth, and depth becomes a value
you pick at inference time — per token, if you use an early-exit test. This explorer
walks the verified mechanics from Huginn (arXiv 2502.05171), Parcae (arXiv 2604.12946 +
the public SandyResearch/parcae code), and LOTUS (arXiv 2606.31779): input re-injection
at every iteration, the ρ(Ā) < 1 stability story, log-normal Poisson depth sampling with
truncated backprop, saturating test-time scaling curves, KV caching along the loop axis,
and a four-line efficiency ledger where looping helps exactly two lines.
