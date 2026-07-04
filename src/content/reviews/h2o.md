---
title: 'H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs'
arxivId: '2306.14048'
authors: 'Zhang et al.'
lab: 'UT Austin / Stanford / CMU'
venue: 'NeurIPS 2023'
date: 2026-06-30
tags: [kv-cache, sparse-attention]
summary: 'A small set of "heavy-hitter" tokens dominates attention mass, so evicting everything else from the KV cache preserves quality at a fraction of the memory.'
links: [kivi, duo-attention]
source: 'manual'
---

## What it does

H2O is one of the founding papers of KV-cache *eviction*: rather than storing keys and values for every past token, it keeps only a small budget of entries and throws the rest away during generation, cutting cache memory to a fraction of the full size with little quality loss on the benchmarks studied.

## How it works

The empirical anchor is that attention mass is extremely skewed: a small subset of tokens — the *heavy hitters* — receives the bulk of attention across steps, and masking them out collapses model quality. H2O turns this into an online eviction policy. Each token accumulates its attention scores over decoding; when the cache budget is exceeded, the entry with the lowest accumulated score is evicted. The retained set is the union of recent tokens and heavy hitters. The paper frames budgeted retention as a dynamic submodular maximization problem, which gives the greedy accumulated-score policy a theoretical footing under idealized assumptions. Because eviction shrinks the working set, it composes with serving stacks to raise batch size and throughput substantially over full-cache baselines.

## Why it matters

H2O established the vocabulary — heavy hitters, cache budget, eviction policy — that the entire KV-compression subfield now uses, and demonstrated that the cache is *redundant* in a structured, exploitable way. It is the natural sparsity-side complement to quantization approaches like KIVI: one drops entries, the other shrinks them.

## Open questions

- Accumulated attention is a backward-looking statistic; tokens that become important *later* (needle-in-a-haystack retrieval) can be evicted before they matter, and long-context follow-ups have documented exactly this failure mode.
- Scores are head- and layer-heterogeneous; a single global policy is likely leaving quality on the table.
- How eviction interacts with quantized caches, and with diffusion-style decoding where "past" is not a fixed prefix, both remain open.
