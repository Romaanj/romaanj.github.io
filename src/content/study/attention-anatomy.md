---
title: 'Attention-head anatomy: MHA → MQA → GQA → MLA'
title_ko: 'Attention head 해부: MHA → MQA → GQA → MLA'
track: models
summary: 'One dial from private K/V pairs to a single shared latent — how each head-sharing scheme rewires attention and what it does to KV-cache bytes, with verified configs from Llama, Mistral, and DeepSeek.'
summary_ko: 'head마다 따로 갖던 K/V pair가 하나의 shared latent로 줄어드는 과정 — 각 head-sharing 방식이 attention 배선과 KV cache byte를 어떻게 바꾸는지, Llama·Mistral·DeepSeek의 검증된 config로 따라간다.'
date: 2026-07-05
order: 3
interactive: /study/models/attention-anatomy/
tags: [attention, kv-cache, architecture]
---

MHA, MQA, GQA, and MLA are usually presented as four separate designs, but they are
one decision viewed from four points: how many query heads share each cached K/V.
This interactive note wires up all four side by side — hover a query head to see
exactly which K/V it reads — then puts real model configs (Llama-2-7B, Llama-3-8B,
Mistral-7B, DeepSeek-V2/V3) through a KV-bytes calculator, and opens up DeepSeek's
Multi-head Latent Attention matrix by matrix: the 512-dim latent, weight absorption,
and the decoupled 64-dim RoPE key that makes it work.
