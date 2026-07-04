---
title: 'KV-cache quantization, byte by byte'
title_ko: 'KV cache quantization, byte 단위로'
track: quantization
summary: 'Follow one key vector from fp16 to packed low-bit integers and back — scales, zero-points, grouping, and where the reconstruction error actually lands.'
summary_ko: 'key vector 하나를 fp16에서 packed low-bit integer로, 다시 원래대로 따라간다 — scale, zero-point, grouping, 그리고 reconstruction error가 실제로 어디에 떨어지는지.'
date: 2026-07-04
order: 1
interactive: /study/quantization/kv-quant-dataflow/
tags: [kv-cache, quantization]
---

The fastest way to demystify KV-cache quantization is to follow a single key vector
through the whole pipeline: group it, find its scale and zero-point, round it into a
handful of bits, pack it, then dequantize it at attention time and watch the error
show up in the attention scores. This interactive note draws every step of that data flow.
