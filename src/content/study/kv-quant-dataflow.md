---
title: 'KV-cache quantization, byte by byte'
track: quantization
summary: 'Follow one key vector from fp16 to packed low-bit integers and back — scales, zero-points, grouping, and where the reconstruction error actually lands.'
date: 2026-07-04
order: 1
interactive: /study/quantization/kv-quant-dataflow/
tags: [kv-cache, quantization]
---

The fastest way to demystify KV-cache quantization is to follow a single key vector
through the whole pipeline: group it, find its scale and zero-point, round it into a
handful of bits, pack it, then dequantize it at attention time and watch the error
show up in the attention scores. This interactive note draws every step of that data flow.
