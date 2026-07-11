---
title: 'RoPE: sixty-four dials on every attention head'
title_ko: 'RoPE: 모든 attention head에 달린 64개의 다이얼'
track: models
summary: 'Where rotary position embeddings actually land in the attention path, what each frequency dial does, how per-band cosines compose into attention logits, and how PI, NTK-aware scaling, YaRN and base-scaling each edit the dial ladder for long context — with verified configs and a worked d = 8 example.'
summary_ko: 'RoPE가 attention 경로의 정확히 어디에 꽂히는지, frequency 다이얼 하나하나가 무엇을 하는지, band별 cosine이 attention logit으로 어떻게 합성되는지, 그리고 PI·NTK-aware·YaRN·base-scaling이 long context를 위해 다이얼 사다리를 어떻게 다르게 수정하는지 — 검증된 config와 d = 8 수치 예제로 따라간다.'
date: 2026-07-07
order: 5
interactive: /study/models/rope/
tags: [attention, positional-encoding, architecture]
---

RoPE never touches the values and adds nothing to the residual stream: it rotates each
query/key channel pair by an angle proportional to token position, at d/2 geometrically
spaced speeds. This interactive note pins down the exact insertion point (post-projection,
post-QK-norm, q and k only — with the "why not v?" answer), spins a single 2×2 block live
including both pairing conventions and the checkpoint-compatibility warning, animates all
64 dials as a wall of clocks with their wavelengths, composes per-band cosines into a
logit-vs-distance curve tied to Barbero et al.'s trained-Gemma measurements, and puts PI,
NTK-aware scaling, YaRN's ramp (α = 1, β = 32) and Llama-3's θ = 500,000 route on one
shared wavelength axis — plus partial RoPE, a systems corner (rotate-at-write caches,
MLA's decoupled key, KVQuant's pre-RoPE quantization), and a machine-checked d = 8
worked example.
