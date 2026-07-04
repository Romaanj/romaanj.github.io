---
title: 'Hybrid attention: inside Qwen3.5''s linear + full stack'
title_ko: '하이브리드 어텐션: Qwen3.5의 linear + full 스택 해부'
track: models
summary: 'A tensor-by-tensor tear-down of Qwen3.5''s hybrid architecture — the 3:1 Gated DeltaNet / Gated Attention interleave, both token mixers side by side, and the memory race between a growing KV cache and a constant recurrent state.'
summary_ko: 'Qwen3.5 하이브리드 아키텍처를 tensor 단위로 해부 — 3:1 Gated DeltaNet / Gated Attention 인터리브, 두 token mixer 비교, 그리고 증가하는 KV cache와 고정 크기 recurrent state의 메모리 경주.'
date: 2026-07-05
order: 2
interactive: /study/models/hybrid-attention/
tags: [hybrid-architecture, linear-attention, kv-cache]
---

Qwen3.5 keeps full softmax attention in only one of every four layers; the other three
replace it with Gated DeltaNet, a linear mixer whose memory is a fixed 128×128 state
matrix per head instead of an ever-growing KV cache. This interactive tear-down walks
the verified 64-layer config, opens both mixers with every tensor shape labeled, and
races the hybrid's memory footprint against an all-full-attention counterfactual out
to the native 262K-token context — crossover arithmetic included.
