---
title: 'SDAR: anatomy of a block-diffusion LM'
track: models
summary: 'A block-by-block tear-down of SDAR — how a block-diffusion language model mixes autoregressive structure across blocks with parallel denoising inside them.'
date: 2026-07-04
order: 1
interactive: /study/models/sdar/
tags: [diffusion-llm, architecture]
---

SDAR sits between two worlds: blocks are generated left-to-right like an autoregressive
model, but the tokens inside each block are denoised in parallel like a diffusion model.
This tear-down walks the full forward pass — embeddings, block-wise attention masks,
the denoising loop, and the KV cache that makes it fast — as an interactive diagram.
