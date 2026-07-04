---
title: 'LLaDA: Large Language Diffusion Models'
arxivId: '2502.09992'
authors: 'Nie et al.'
lab: 'RUC GSAI / Ant Group'
venue: 'ICML 2025'
date: 2026-06-28
tags: [diffusion-llm]
summary: 'An 8B masked-diffusion language model trained from scratch that matches strong autoregressive baselines, showing diffusion is a viable alternative recipe for LLMs.'
links: [fast-dllm]
source: 'manual'
---

## What it does

LLaDA is the first large-scale demonstration that a *diffusion* objective — not next-token prediction — can train a general-purpose language model at the 8B scale. The model is pretrained from scratch on trillions of tokens, then instruction-tuned, and ends up competitive with strong autoregressive baselines of similar size on in-context learning and instruction following.

## How it works

Instead of factorizing the sequence left-to-right, LLaDA defines a forward process that progressively *masks* tokens at a random ratio, and a reverse process that recovers them. A standard Transformer (without a causal mask) acts as the mask predictor: given a partially masked sequence, it predicts all masked positions simultaneously. Training optimizes a likelihood bound over random masking ratios, which makes the model a principled generative model rather than a heuristic denoiser. At inference, generation proceeds by iterative demasking — start from a fully masked response and progressively commit tokens over a number of steps, with remasking strategies controlling which predictions are kept per step.

## Why it matters

The autoregressive factorization has been treated as almost synonymous with "LLM." LLaDA shows key LLM capabilities — scaling behavior, in-context learning, instruction following — come from the generative-modeling principle, not from left-to-right decoding specifically. The bidirectional formulation also sidesteps some AR pathologies: the paper highlights reversal-style tasks where left-to-right models struggle. For efficiency research, dLLMs open a new axis: quality is traded against the *number of refinement steps*, not just tokens per second.

## Open questions

- Vanilla dLLM decoding recomputes full bidirectional attention every step, so out of the box it is much slower than a KV-cached AR model — the acceleration story (caching, parallel decoding) is where follow-up work lives.
- How masking-ratio schedules and remasking policies interact with reasoning quality is still poorly characterized.
- Whether dLLMs keep pace with AR models as both scale beyond 8B remains open.
