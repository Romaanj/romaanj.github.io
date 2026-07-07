---
title: 'Why 8× smaller can load slower: mixed-precision KV caches inside the kernel'
title_ko: '8× 작은데 왜 더 느리게 읽히는가: kernel 안의 mixed-precision KV cache'
track: gpu
summary: 'Follow packed low-bit KV bytes through HBM, shared memory, and the unpack/dequant instruction stream — why naive 4-bit lost to 8-bit, what mixed widths break, and the published kernel patterns that fix it.'
summary_ko: 'packed low-bit KV byte를 HBM, shared memory, 그리고 unpack/dequant instruction stream까지 따라간다 — naive 4-bit가 8-bit에 진 이유, 혼합 width가 깨뜨리는 것들, 그리고 그것을 고치는 공개된 kernel 패턴.'
date: 2026-07-07
order: 2
interactive: /study/gpu/mixed-precision-kernel/
tags: [kv-cache, quantization, gpu, kernels]
---

Byte counting says a 2-bit cache should load 8× faster than fp16. The kernel disagrees:
every packed value must be unpacked, dequantized, and placed in the right lane before a
single FMA fires, and all of that runs on CUDA cores against a hard ops-per-byte budget.
This interactive note walks the whole path — coalesced transactions, lop3/prmt bit tricks,
the metadata second stream, the three ways mixed widths break warp execution, and the
published fix patterns from MARLIN, FLUTE, QServe, KVQuant, and DiffKV — ending with an
honest map of when low-bit actually wins.
