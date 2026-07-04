---
title: 'Fast-dLLM: Training-Free Acceleration of Diffusion LLMs via KV Cache and Parallel Decoding'
arxivId: '2505.22618'
authors: 'Wu et al.'
lab: 'NVIDIA / MIT / HKU'
date: 2026-07-04
tags: [diffusion-llm, kv-cache]
summary: 'A block-wise approximate KV cache plus confidence-thresholded parallel decoding makes open diffusion LLMs an order of magnitude faster with almost no accuracy loss — no retraining needed.'
links: [llada, kivi]
source: 'manual'
---

## What it does

Fast-dLLM attacks the two reasons open diffusion LLMs (LLaDA, Dream) decode slowly in practice: bidirectional attention prevents the standard KV cache, and decoding multiple tokens per step naively degrades quality. It fixes both *training-free*, reporting order-of-magnitude end-to-end throughput gains (up to roughly 27× in the best configurations) at near-baseline accuracy.

## How it works

Two components. First, a **block-wise approximate KV cache**: generation proceeds block by block, and within a block the keys/values of tokens outside the active block change little between refinement steps — the paper verifies this by measuring cosine similarity of KV activations across adjacent steps. So Fast-dLLM caches them and refreshes the cache only at block boundaries. A DualCache variant caches both the prefix and the still-masked suffix tokens, since the suffix is static within a block too. Second, **confidence-aware parallel decoding**: rather than committing a fixed number of tokens per step, it commits exactly those whose predicted confidence exceeds a threshold. The authors give a supporting argument that when confidences are high enough, greedy parallel decoding agrees with greedy sequential decoding — connecting the heuristic to a correctness condition rather than leaving it purely empirical.

## Why it matters

This is the paper that made dLLM inference *feel* competitive: caching plus adaptive parallelism recovers most of the speed advantage AR models get from their KV cache, without touching the weights. It also set the template for a now-busy subfield — treat "how stale can cached states get?" and "how many tokens can commit at once?" as measurable, tunable quantities in the accuracy-throughput trade.

## Open questions

- The cache is *approximate*; staleness error is controlled by block length but the paper leaves the accuracy-staleness frontier largely uncharted.
- Confidence thresholds are global — per-position or per-step-adaptive policies are an obvious next lever.
- How approximate dLLM caches interact with cache *compression* (quantization, eviction) developed for AR models is an open composition question.
