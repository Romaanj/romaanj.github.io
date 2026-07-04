---
title: 'DuoAttention: Efficient Long-Context Inference with Retrieval and Streaming Heads'
arxivId: '2410.10819'
authors: 'Xiao et al.'
lab: 'MIT-HAN Lab'
venue: 'ICLR 2025'
date: 2026-07-01
tags: [sparse-attention, kv-cache, serving]
summary: 'Only a minority of attention heads actually retrieve from long context; giving full KV cache to those and a tiny constant cache to the rest cuts memory without hurting retrieval.'
links: [h2o, kivi, gated-deltanet]
source: 'manual'
---

## What it does

DuoAttention reduces long-context KV memory and latency by splitting attention heads into two functional classes: **retrieval heads**, which genuinely need access to the full context, and **streaming heads**, which mostly attend to recent tokens and attention sinks. Full KV cache is kept only for retrieval heads; streaming heads get a constant-size cache of sinks plus a recent window.

## How it works

The classification is the interesting part. Instead of ranking heads by attention-score profiles (which can mislead), DuoAttention *learns* the split: a gate per head interpolates between full attention and streaming attention, and the gates are optimized on synthetic passkey-retrieval data with a distillation-style objective — match the full-cache model's outputs while pushing gates toward the cheap streaming mode. Heads whose gates stay high are the ones the output measurably depends on for long-range retrieval. At deployment the gate values are thresholded into a fixed binary head assignment, so inference uses two cache layouts side by side with no per-step decision cost. The method is training-light (only gates are learned), model-agnostic, and composes with quantization for further memory reduction.

## Why it matters

This paper moved KV compression from *token-level* selection (which tokens to keep, as in eviction work) to *head-level* structure (which subcomputations need long context at all). The optimization-based identification is a genuinely better measurement instrument than attention-score inspection, and the resulting memory reduction is structural rather than input-dependent — friendlier to serving systems that need predictable footprints. It also gives mechanistic backing to hybrid-architecture intuitions: if most heads are streaming anyway, architectures that replace them with cheap layers have headroom.

## Open questions

- The split is calibrated on passkey-style retrieval; whether the identified retrieval-head set is task-universal (reasoning, code, multi-hop) is not fully settled.
- A binary head split is coarse — a budgeted continuum might dominate it.
- How the retrieval/streaming decomposition transfers to GQA/MLA layouts and to bidirectional diffusion decoders is open.
