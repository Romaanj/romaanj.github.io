---
title: 'KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization'
arxivId: '2401.18079'
authors: 'Hooper et al.'
lab: 'UC Berkeley (SqueezeAILab)'
venue: 'NeurIPS 2024'
date: 2026-07-04
tags: [kv-cache, quantization, serving]
topic: kv-cache
summary: 'Pre-RoPE per-channel key and per-token value quantization with sensitivity-weighted non-uniform datatypes plus ~1% fp16 outliers pushes the KV cache to 3 bits at under 0.1 perplexity cost, reaching million-token contexts on a single A100.'
summary_ko: 'Pre-RoPE per-channel key / per-token value 양자화에 sensitivity 가중 non-uniform datatype과 ~1% fp16 outlier 분리를 더해 KV cache를 3-bit(<0.1 PPL 손실)까지 압축, 단일 A100에서 1M-token context를 달성한 논문.'
links: [kivi, h2o, duo-attention]
resources:
  - { label: arXiv, url: 'https://arxiv.org/abs/2401.18079' }
  - { label: PDF, url: 'https://arxiv.org/pdf/2401.18079' }
  - { label: GitHub, url: 'https://github.com/SqueezeAILab/KVQuant' }
analysis:
  ko:
    background: 'Long-context LLM inference에서 KV cache는 sequence length와 batch에 비례해 커져, 긴 context에서는 weight가 아니라 KV cache가 GPU memory의 지배적 병목이 된다.'
    problem: 'Million-token 급 context를 감당하려면 KV cache를 4-bit 이하 초저정밀도로 압축해야 하는데, 정확도 손실 없이 이를 달성하는 방법이 문제다.'
    prior_limits: 'ATOM·FlexGen 같은 기존 방식은 uniform per-token 4-bit 양자화를 KV에 그대로 적용해, key의 outlier channel 구조와 RoPE가 채널을 섞는 효과를 무시하므로 4-bit 미만에서 perplexity가 크게 무너진다.'
    goal: '3-bit(나아가 2-bit) KV cache 양자화를 **<0.1** perplexity 열화로 달성해, 단일 A100-80GB에서 **1M**, 8-GPU에서 **10M** token context를 가능하게 하는 것이 목표다.'
    method: 'Key는 RoPE 적용 전(pre-RoPE)에 per-channel로, Fisher information 가중 k-means로 offline calibration(Wikitext-2 16 샘플)한 non-uniform signpost(nuqX)로 양자화하고, Value는 per-token으로 online 양자화하며, per-vector threshold로 ~1% outlier를 fp16 sparse 포맷으로 분리하고 attention sink인 첫 token은 fp16으로 유지, dequant 후 RoPE를 적용하는 custom CUDA kernel로 추론한다.'
    key_idea: '양자화 축을 tensor의 분포 구조에 맞추는 것 — key의 outlier는 RoPE 이전에 channel 방향으로 정렬돼 있으므로 pre-RoPE per-channel + sensitivity 가중 non-uniform datatype으로 잡고, value는 per-token으로 잡으며, 남는 소수 outlier만 sparse로 뺀다.'
    validation: 'LLaMA-7B/13B/30B/65B, Llama-2, Llama-3, Mistral-7B에 대해 Wikitext-2·C4 perplexity로 평가하고, passkey retrieval과 long-context 실험, A100 kernel microbenchmark로 검증했다.'
    results: '4-bit **<0.02**, 3-bit **<0.1**, 2-bit **<0.5** perplexity 열화(각각 **3.7×/4.8×/6.9×** 압축), LLaMA-7B 기준 단일 A100-80GB에서 **1M** token·8-GPU에서 **10M** token context, kernel은 fp16 대비 **1.2–1.7×** speedup을 보였다.'
    comparison: 'Uniform INT4/INT3/INT2, NormalFloat NF4/NF3/NF2, ATOM(group 128), FlexGen(group 64)을 일관되게 상회하며, 동시기 2-bit 방법인 KIVI와 같은 per-channel key 관찰을 공유한다.'
    significance: 'Per-channel key / per-token value 비대칭과 outlier 분리라는 KV 양자화의 표준 레시피를 calibration 기반 non-uniform datatype과 함께 정립해, 고정된 하드웨어에서 context 길이를 수백만 token 단위로 확장할 수 있음을 보였다.'
    limitations: '논문 스스로 offline calibration이 out-of-distribution 입력에 완전히 일반화되지 않을 수 있음과 per-vector outlier 처리의 부가 memory overhead를 언급하며, (리뷰어 판단) 보고된 speedup은 kernel 수준이라 paged fp16 serving 대비 end-to-end throughput 이득은 별도 검증이 필요하다.'
    future_work: '논문에 명시적 future work 절은 없으며, (리뷰어 판단) GQA/MLA 구조로의 이식, eviction·streaming 기법과의 결합, serving 스택 통합이 자연스러운 다음 단계다.'
    resources: 'Calibration 코드와 custom CUDA kernel을 포함한 공식 구현이 GitHub(SqueezeAILab/KVQuant)에 공개돼 있다.'
  en:
    background: 'In long-context LLM inference the KV cache grows linearly with sequence length and batch size, so at long contexts it, not the weights, becomes the dominant GPU-memory bottleneck.'
    problem: 'Serving million-token contexts requires compressing the KV cache below 4 bits, and the problem is doing so without sacrificing accuracy.'
    prior_limits: 'Prior approaches such as ATOM and FlexGen apply uniform per-token 4-bit quantization to the KV cache, ignoring the outlier-channel structure of keys and the channel-mixing effect of RoPE, so perplexity collapses below 4 bits.'
    goal: 'Achieve 3-bit (and even 2-bit) KV cache quantization with **<0.1** perplexity degradation, enabling **1M**-token context on a single A100-80GB and **10M** tokens on an 8-GPU system.'
    method: 'Keys are quantized per-channel before RoPE using non-uniform signposts (nuqX) derived offline via Fisher-information-weighted k-means on a 16-sample Wikitext-2 calibration set, values are quantized per-token online, roughly 1% of outliers are isolated into an fp16 sparse format via per-vector thresholds, the attention-sink first token stays in fp16, and custom CUDA kernels apply RoPE after dequantization at inference.'
    key_idea: 'Align the quantization axis with the tensor distribution — key outliers are channel-aligned before RoPE, so quantize keys per-channel pre-RoPE with sensitivity-weighted non-uniform datatypes, values per-token, and push the few remaining outliers into a tiny sparse store.'
    validation: 'Evaluated via Wikitext-2 and C4 perplexity on LLaMA-7B/13B/30B/65B, Llama-2, Llama-3, and Mistral-7B, plus passkey-retrieval and long-context experiments and A100 kernel microbenchmarks.'
    results: 'Perplexity degradation of **<0.02** at 4-bit, **<0.1** at 3-bit, and **<0.5** at 2-bit (**3.7×/4.8×/6.9×** compression), context lengths of **1M** tokens for LLaMA-7B on one A100-80GB and **10M** on 8 GPUs, and **1.2–1.7×** kernel speedups over fp16.'
    comparison: 'Consistently beats uniform INT4/INT3/INT2, NormalFloat NF4/NF3/NF2, ATOM (group size 128), and FlexGen (group size 64), and shares the per-channel-key observation with the concurrent 2-bit method KIVI.'
    significance: 'It codified the per-channel-key / per-token-value asymmetry plus outlier decomposition as the standard KV-quantization recipe, and showed calibration-based non-uniform datatypes stretch context length to millions of tokens on fixed hardware.'
    limitations: 'The paper itself notes that offline calibration may not generalize to out-of-distribution inputs and that per-vector outlier handling adds some memory overhead, and (reviewer judgment) the reported speedups are kernel-level, so end-to-end throughput gains versus paged fp16 serving remain to be shown.'
    future_work: 'No explicit future-work section is stated in the paper, and (reviewer judgment) natural next steps are transfer to GQA/MLA architectures, combination with eviction and streaming methods, and full serving-stack integration.'
    resources: 'The official implementation, including calibration code and custom CUDA kernels, is public on GitHub (SqueezeAILab/KVQuant).'
source: 'manual'
---

## Notes

KVQuant and KIVI landed within days of each other and converged on the same core observation — key outliers are per-channel, values are not — but KVQuant goes further on the datatype side: pre-RoPE quantization, Fisher-weighted non-uniform signposts, and a dense-and-sparse outlier split, at the cost of an offline calibration pass that KIVI avoids. Its headline framing (context length on fixed hardware, rather than accuracy at a fixed budget) made KV memory, not compute, the explicit target, and most later KV-quantization work is measured against the recipe it standardized. The kernel-level 1.2–1.7× numbers are honest but narrow; the end-to-end serving story was left to later systems.

KIVI가 calibration-free 노선이라면 KVQuant는 offline calibration으로 datatype 자체를 최적화하는 노선으로, 두 논문이 KV 양자화 설계 공간의 양 극단을 정의한다.
