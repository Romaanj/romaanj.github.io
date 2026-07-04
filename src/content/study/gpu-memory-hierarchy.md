---
title: 'Where the bytes live: GPU memory hierarchy'
title_ko: 'byte는 어디에 사는가: GPU memory hierarchy'
track: gpu
summary: 'Registers, shared memory, L2, HBM — an interactive map of the GPU memory hierarchy and why inference kernels spend their lives waiting on the slowest tier.'
summary_ko: 'Register, shared memory, L2, HBM — GPU memory hierarchy의 인터랙티브 지도, 그리고 inference kernel이 왜 가장 느린 계층을 기다리며 대부분의 시간을 보내는지.'
date: 2026-07-04
order: 1
interactive: /study/gpu/memory-hierarchy/
tags: [gpu, roofline]
---

Every performance question about inference eventually becomes a question about where
the bytes live. This interactive note maps the GPU memory hierarchy — registers, shared
memory, L2, and HBM — with the bandwidth and capacity of each tier, and shows why
decode-time kernels are memory-bound long before they run out of FLOPs.
