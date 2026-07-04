---
title: 'Gated Delta Networks: Improving Mamba2 with Delta Rule'
arxivId: '2412.06464'
authors: 'Yang et al.'
lab: 'MIT / NVIDIA'
venue: 'ICLR 2025'
date: 2026-07-02
tags: [linear-attention, kv-cache]
summary: 'Combining Mamba2-style gated decay with the delta-rule targeted memory update yields a linear-attention layer that beats both parents, especially in hybrid stacks.'
links: [duo-attention]
source: 'manual'
---

## What it does

Gated DeltaNet is a linear-attention / state-space layer that merges two previously separate memory-control mechanisms: the **gating** (data-dependent decay) that makes Mamba2 good at forgetting stale context, and the **delta rule** update that makes DeltaNet good at precise, targeted memory replacement. The combined layer outperforms both parents on language modeling, retrieval-style probes, and length extrapolation, and slots into hybrid architectures alongside sliding-window attention or Mamba2 layers.

## How it works

Linear-attention models maintain a fixed-size matrix state instead of a growing KV cache. Pure decay-based updates (Mamba2) fade *everything* uniformly — good hygiene, poor precision, since you cannot erase one association without eroding the rest. The delta rule instead reads out the current value bound to a key and writes back only the difference toward the new value — precise editing, but with no mechanism for global forgetting. Gated DeltaNet's update applies a data-dependent scalar decay to the state *and then* a delta-rule correction, so the layer can both fade history and surgically overwrite specific associations. The authors derive a chunkwise-parallel training algorithm (in the WY-representation family used for DeltaNet) that keeps the layer hardware-efficient on tensor cores despite the sequential-looking recurrence.

## Why it matters

For efficient inference, constant-state layers are the endgame answer to KV-cache growth: memory per sequence is O(1) in context length. This paper is a clean demonstration that the *form of the state update* — not just state size — determines retrieval ability, and that decay and targeted replacement are complementary rather than competing. The hybrid results also align neatly with head-level findings like DuoAttention: keep a little full attention where retrieval truly needs it, make everything else cheap.

## Open questions

- Fixed-size state still imposes a hard capacity ceiling; where exactly hybrid models must fall back to full attention is an empirical, task-dependent boundary.
- The gating/delta decomposition invites interpretability questions: can we read out *what* the state chose to overwrite?
- Whether these layers can serve as drop-in decoders for diffusion-style (bidirectional) LMs is essentially unexplored.
