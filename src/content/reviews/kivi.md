---
title: 'KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache'
arxivId: '2402.02750'
authors: 'Liu et al.'
lab: 'Rice University'
venue: 'ICML 2024'
date: 2026-06-29
tags: [quantization, kv-cache]
topic: 'kv-cache'
summary: 'Asymmetric KV quantization derived from element-distribution analysis — per-channel keys, per-token values — pushes the KV cache to tuning-free 2 bits, cutting peak memory 2.6x and lifting serving throughput up to 3.47x.'
summary_ko: 'KV cache 원소 분포 분석에서 도출한 비대칭 양자화(K per-channel, V per-token)로 tuning 없이 2bit KV cache를 달성, peak memory 2.6배 절감과 최대 3.47배 throughput 향상을 얻는다.'
links: [kvquant, h2o, duo-attention]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2402.02750' }
  - { label: 'GitHub', url: 'https://github.com/jy-yuan/KIVI' }
analysis:
  ko:
    background: 'LLM serving은 batching으로 요청당 비용을 낮추는데, batch가 커지고 context가 길어지면 weight가 아니라 KV cache가 메모리 상한이 되고, KV cache 로딩 동안 연산 코어가 놀게 되는 memory-bound decode가 속도를 제한한다.'
    problem: 'KV cache를 4bit 아래의 극저비트(2bit)로 줄이면서도 fine-tuning이나 calibration 없이 정확도를 유지하는 방법이 없었다.'
    prior_limits: '기존 KV quantization은 K와 V를 같은 방식(주로 per-token uniform)으로 다뤄 4bit 밑에서 급격히 붕괴했고(FlexGen의 4bit group-wise, ATOM 계열이 참조점), KV cache 원소 분포 자체에 대한 심층 분석이 부재했다.'
    goal: 'FP16과 거의 같은 품질을 유지하는 tuning-free, plug-in 방식의 2bit KV cache quantizer로 더 큰 batch와 더 높은 throughput을 확보하는 것.'
    method: 'Key cache는 per-channel, value cache는 per-token으로 2bit 양자화하되(group size **32**), 최근 **128** 토큰의 residual window를 full precision으로 유지하다가 그룹 단위로 양자화 store에 편입시키고, dequantization을 tiled matmul에 fuse한 hardware-friendly attention 구현으로 streaming decoding과 호환시킨다.'
    key_idea: 'K와 V는 통계적으로 다른 객체라는 관측 — key는 특정 채널에 지속적 outlier magnitude가 몰려 있어 channel 방향으로 묶어 양자화해야 하고, value는 채널 구조가 없는 대신 attention output 단위로 오차가 작동하므로 token 방향으로 양자화해야 한다는 비대칭 처방이다.'
    validation: 'Llama/Llama-2(7B·13B), Falcon-7B, Mistral-7B를 LM-Eval(CoQA, TruthfulQA, GSM8K)과 LongBench(Qasper, QMSum, TREC, SAMSum, LCC, RepoBench-P 등)에서 평가하고 실제 serving workload로 throughput을 측정했다.'
    results: '대부분의 태스크에서 FP16에 근접한 품질을 유지하면서 weight 포함 peak memory를 **2.6배** 줄이고, 최대 **4배** 큰 batch와 **2.35~3.47배** throughput 향상을 달성했다.'
    comparison: 'FP16, INT4/INT2 per-token 및 per-channel uniform baseline과 비교해 대칭적 2bit per-token은 정확도가 붕괴하는 반면 KIVI의 비대칭 분할은 2bit에서 품질을 유지한다.'
    significance: '이후 거의 모든 KV-cache quantization 연구가 계승하는 K/V 비대칭(per-channel K, per-token V)을 처음 명시적으로 정립했고, 2bit KV cache를 실제 serving 옵션으로 만들었으며, tensor 분포 측정에서 method가 직접 도출되는 모범 사례다.'
    limitations: '논문 자체가 multi-query attention인 Falcon-7B는 2bit에서 열화가 커 INT4가 필요하다고 보고하며, (리뷰어 판단) 128-token full-precision residual window가 실질적 정확도 기여를 하는데 그 기여도 분해는 충분히 분석되지 않았다.'
    future_work: '논문은 prefill·decoding 단계의 quantization overhead 축소를 향후 과제로 들고, (리뷰어 판단) GQA/MLA 아키텍처로의 outlier-channel 구조 전이 여부와 eviction 계열 방법과의 결합은 열린 문제다.'
    resources: '공식 코드가 GitHub(jy-yuan/KIVI)에 공개되어 있으며, tuning-free plug-in 방식이라 별도 checkpoint 없이 기존 모델에 바로 적용 가능하다.'
  en:
    background: 'LLM serving amortizes cost via batching, but at large batch and long context the KV cache — not the weights — becomes the memory ceiling, and loading it leaves the compute cores idle, making memory-bound decode the speed limiter.'
    problem: 'There was no way to push the KV cache to extreme low bit-width (2-bit, below 4-bit) without fine-tuning or calibration while preserving accuracy.'
    prior_limits: 'Prior KV quantization treated K and V identically (mostly per-token uniform) and collapsed sharply below 4 bits — FlexGen-style 4-bit group-wise and ATOM-type schemes being the reference points — with no in-depth study of KV cache element distributions.'
    goal: 'A tuning-free, plug-in 2-bit KV cache quantizer that keeps near-FP16 quality while unlocking larger batches and higher serving throughput.'
    method: 'Quantize the key cache per-channel and the value cache per-token to 2 bits (group size **32**), keep the most recent **128** tokens as a full-precision residual window that is folded into the quantized store group-by-group, and fuse dequantization into tiled matmul so the attention kernel stays compatible with streaming decoding.'
    key_idea: 'K and V are statistically different objects — keys carry persistent outlier channels of large magnitude (so group and quantize along the channel dimension), while value errors act per attention output (so quantize per token) — an asymmetric prescription read directly off the tensor statistics.'
    validation: 'Evaluated Llama/Llama-2 (7B, 13B), Falcon-7B, and Mistral-7B on LM-Eval tasks (CoQA, TruthfulQA, GSM8K) and LongBench (Qasper, QMSum, TREC, SAMSum, LCC, RepoBench-P, etc.), plus throughput measurements on a real serving workload.'
    results: 'Maintains near-FP16 quality on most tasks while cutting peak memory (model weights included) by **2.6x**, enabling up to **4x** larger batch sizes and **2.35-3.47x** higher throughput.'
    comparison: 'Against FP16 and INT4/INT2 per-token and per-channel uniform baselines, symmetric 2-bit per-token quantization collapses in accuracy whereas the asymmetric KIVI split holds at 2 bits.'
    significance: 'It established the K/V asymmetry (per-channel K, per-token V) that nearly all subsequent KV-cache quantization work inherits, made a 2-bit KV cache a practical serving option, and is a model case of a method falling directly out of measured tensor distributions.'
    limitations: 'The paper itself reports that Falcon-7B (multi-query attention) degrades notably at 2 bits and needs INT4 for acceptable accuracy, and (reviewer judgment) the 128-token full-precision residual window does real accuracy work whose per-task contribution is under-analyzed.'
    future_work: 'The paper names reducing quantization overhead in the prefill and decoding phases as future work, and (reviewer judgment) whether the outlier-channel structure transfers to GQA/MLA caches and how KIVI composes with eviction methods remain open.'
    resources: 'Official code is released on GitHub (jy-yuan/KIVI); being tuning-free and plug-in, it applies to existing checkpoints with no extra artifacts.'
source: 'manual'
---

## Notes

KIVI is the reference point for asymmetric KV-cache quantization: it turned a distributional observation — structured outlier channels in keys, none in values — into a quantization axis choice, and nearly every later KV-quant paper either inherits the per-channel-K/per-token-V split or defines itself against it. The full-precision residual window it introduced is now a standard ingredient, quietly doing accuracy work in most streaming low-bit schemes. Read alongside KVQuant for the non-uniform/outlier-isolation branch of the same problem, and H2O / DuoAttention for the orthogonal axis of dropping cache entries rather than shrinking them.

KIVI는 "tensor를 먼저 측정하고 method는 그 따름정리로 나온다"는 전개의 교과서적 사례로, KV quantization 계열을 읽을 때 출발점으로 삼기 좋은 논문이다.
