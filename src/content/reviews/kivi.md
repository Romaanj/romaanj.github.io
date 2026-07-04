---
title: 'KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache'
arxivId: '2402.02750'
authors: 'Liu et al.'
lab: 'Rice University'
venue: 'ICML 2024'
date: 2026-06-29
tags: [quantization, kv-cache]
summary: 'Per-channel key quantization plus per-token value quantization pushes the KV cache to 2 bits with no tuning, unlocking much larger batches.'
links: [h2o, duo-attention]
source: 'manual'
---

## What it does

KIVI compresses the KV cache of autoregressive LLMs down to 2 bits per element, with no fine-tuning or calibration of the model itself. The point is serving economics: KV cache, not weights, is what caps batch size at long context, so shrinking it directly buys throughput.

## How it works

The core contribution is an *asymmetry observation* from studying element distributions in the cache. Key states have a few channels with persistently large magnitudes — outliers are structured **per channel** — so keys should be quantized along the channel dimension, grouping elements that share a scale. Value states show no such channel structure, but errors in values matter per attention *output*, which argues for **per-token** quantization. KIVI applies exactly this split: per-channel keys, per-token values. Because per-channel scales cannot be finalized for tokens that are still arriving, a small residual window of the most recent tokens is kept in full precision and folded into the quantized store in groups. The streaming design keeps the attention computation compatible with standard decoding.

## Why it matters

KIVI made explicit that K and V are statistically different objects and deserve different quantizers — a fact nearly every later KV-quantization paper builds on. Practically, a 2-bit cache means several-fold less memory per sequence, which translates into proportionally larger batches and real end-to-end throughput gains on memory-bound decode. It is also a clean example of "measure the tensor first": the method falls directly out of the distributional analysis.

## Open questions

- The full-precision residual window is doing real work; how small it can get, and what its accuracy contribution is per task, deserves more scrutiny.
- Outlier-channel structure varies across architectures (GQA, MLA) — does the asymmetry argument transfer intact?
- Interaction with eviction methods (quantize-then-evict vs evict-then-quantize) is largely unexplored territory.
