---
title: 'Where the bytes live: GPU memory hierarchy'
track: gpu
summary: 'Registers, shared memory, L2, HBM — an interactive map of the GPU memory hierarchy and why inference kernels spend their lives waiting on the slowest tier.'
date: 2026-07-04
order: 1
interactive: /study/gpu/memory-hierarchy/
tags: [gpu, roofline]
---

Every performance question about inference eventually becomes a question about where
the bytes live. This interactive note maps the GPU memory hierarchy — registers, shared
memory, L2, and HBM — with the bandwidth and capacity of each tier, and shows why
decode-time kernels are memory-bound long before they run out of FLOPs.
