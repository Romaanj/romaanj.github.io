---
title: 'DuoAttention: Efficient Long-Context Inference with Retrieval and Streaming Heads'
arxivId: '2410.10819'
authors: 'Xiao et al.'
lab: 'MIT-HAN Lab'
venue: 'ICLR 2025'
date: 2026-07-01
tags: [kv-cache, sparse-attention, serving]
topic: kv-cache
summary: 'Learns which attention heads actually retrieve from long context and gives only those a full KV cache, cutting long-context memory up to 2.55x with retrieval accuracy intact.'
summary_ko: '어떤 attention head가 실제로 long context에서 retrieval을 수행하는지 gate 최적화로 학습해 그 head에만 full KV cache를 주고, 나머지 streaming head는 상수 크기 cache로 줄여 retrieval 성능 손실 없이 최대 2.55배 메모리를 절감한다.'
links: [h2o, kivi, minimax-01, gated-deltanet]
resources:
  - label: arXiv
    url: 'https://arxiv.org/abs/2410.10819'
  - label: GitHub
    url: 'https://github.com/mit-han-lab/duo-attention'
analysis:
  ko:
    background: "Long-context LLM 서빙에서는 모든 attention head의 KV cache가 context 길이에 비례해 커져서 million-token 규모에서는 GPU 메모리를 금방 초과한다."
    problem: "모든 head에 full Key/Value state를 캐싱하면 pre-filling과 decoding 모두 메모리·지연시간에 묶이지만, cache를 단순히 버리면 모델의 long-context retrieval 능력이 무너진다."
    prior_limits: "H2O, TOVA, StreamingLLM 같은 token 단위 eviction은 모든 head가 공유하는 entry를 버려 깊은 context의 retrieval을 망가뜨리고, FastGen 류의 attention-score profiling은 pre-filling에서 FlashAttention과 호환되지 않으며 절감 폭도 제한적이다."
    goal: "입력과 무관한 고정 cache 레이아웃으로 long-context KV 메모리와 지연시간을 구조적으로 줄이되 full attention 수준의 retrieval 정확도를 유지하는 것이 목표다."
    method: "KV head마다 full attention과 streaming(sink+recent) attention을 보간하는 gate α∈[0,1]를 두고 synthetic passkey-retrieval 데이터에서 full-cache 모델 출력에 대한 L2 distillation loss와 L1 sparsity 페널티(λ=0.05)로 **2,000** step만 학습한 뒤 threshold로 이진 head 배정을 확정해, 배포 시 retrieval head는 full KV, streaming head는 sink **16** + recent **64** token의 상수 cache라는 두 정적 레이아웃으로 돌린다."
    key_idea: "핵심은 head 수준의 기능적 이분법 — 소수의 retrieval head만 전체 context가 필요하고 다수의 streaming head는 attention sink와 최근 token만 쓴다는 것 — 을 attention-score 휴리스틱이 아니라 출력 수준 최적화로 판별한다는 점이다."
    validation: "Llama-2-7B(MHA), Llama-3-8B-Instruct-Gradient-1048k, Llama-3-70B, Mistral-7B-v0.2(GQA)에서 최대 1,048K token의 Needle-in-a-Haystack과 LongBench 21개 task, 그리고 MMLU/MBPP/MT-Bench short-context 체크로 검증했다."
    results: "MHA에서 최대 **2.55배** KV 메모리 절감과 **2.18배** decoding 지연 감소(GQA는 **1.67배**/**1.50배**), **1.73배** pre-filling 가속을 달성하면서 KV budget **25%**(MHA)/**50%**(GQA)로 LongBench에서 full attention과 대등하고, QServe의 8-bit weight/4-bit KV quantization과 결합하면 단일 A100에서 **3.3M** token context를 서빙한다(FP16 full attention 대비 **6.4배** capacity)."
    comparison: "H2O, TOVA, StreamingLLM, FastGen과 비교했을 때 token-eviction 계열은 긴 context의 깊은 위치에서 NIAH 정확도가 붕괴하는 반면 DuoAttention만 전 depth에서 정확도를 유지한다."
    significance: "KV 압축을 입력 의존적 token 선택에서 서빙 시스템이 미리 할당할 수 있는 정적 head 구조로 옮겼고, 대부분의 head가 streaming이라는 발견은 hybrid full+linear attention 아키텍처의 기계론적 근거가 된다."
    limitations: "논문 자체가 보고하는 한계는 GQA 모델에서 이득이 줄고(**50%** retrieval 비율 vs MHA **25%**) 모델별 gate 학습 단계가 필요하다는 점이며, passkey로 calibration한 head 분할이 reasoning·code·multi-hop task에도 보편적인지는 완전히 검증되지 않았다(리뷰어 판단)."
    future_work: "논문 내 명시 없음; 이진 분할 대신 budget 기반 연속 head 할당, task 다양성을 반영한 calibration 목적함수, retrieval/streaming 구조를 반영한 GQA/MLA·hybrid 아키텍처 co-design이 자연스러운 다음 단계다(리뷰어 판단)."
    resources: "공식 구현이 MIT-HAN Lab의 GitHub 저장소(mit-han-lab/duo-attention)에 공개되어 있다."
  en:
    background: "Long-context LLM serving is dominated by the KV cache, which grows linearly with context length across every attention head and quickly exceeds GPU memory at million-token scales."
    problem: "Caching full Key/Value states for all heads makes both pre-filling and decoding memory- and latency-bound, yet naively dropping cache entries destroys the model's long-context retrieval ability."
    prior_limits: "Token-level eviction methods (H2O, TOVA, StreamingLLM) discard entries that all heads must share and thus fail deep-context retrieval, while attention-score profiling like FastGen is FlashAttention-incompatible during pre-filling and yields only limited savings."
    goal: "Cut long-context KV memory and latency structurally — with a fixed, input-independent cache layout — while preserving full-attention retrieval accuracy."
    method: "A per-KV-head gate α∈[0,1] interpolates between full and streaming (sink+recent) attention and is trained for only **2,000** steps on synthetic passkey-retrieval data with an L2 distillation loss against the full-cache model plus an L1 sparsity penalty (λ=0.05), after which gates are thresholded into a binary assignment so deployment runs two static layouts — full KV for retrieval heads, a constant **16**-sink + **64**-recent-token cache for streaming heads."
    key_idea: "The key idea is a head-level functional dichotomy — only a minority of retrieval heads need the whole context while the majority of streaming heads use just attention sinks and recent tokens — identified by output-level optimization instead of attention-score heuristics."
    validation: "Validated on Needle-in-a-Haystack up to 1,048K tokens and 21 LongBench tasks, plus MMLU/MBPP/MT-Bench short-context checks, using Llama-2-7B (MHA), Llama-3-8B-Instruct-Gradient-1048k, Llama-3-70B, and Mistral-7B-v0.2 (GQA)."
    results: "Delivers up to **2.55x** KV-memory and **2.18x** decoding-latency reduction on MHA (**1.67x**/**1.50x** on GQA) and **1.73x** faster pre-filling while matching full attention on LongBench at a **25%** (MHA) / **50%** (GQA) KV budget, and combined with QServe 8-bit-weight/4-bit-KV quantization serves a **3.3M**-token context on a single A100 (**6.4x** capacity vs FP16 full attention)."
    comparison: "Against H2O, TOVA, StreamingLLM, and FastGen, DuoAttention is the only method that holds NIAH accuracy across all context depths, where the token-eviction baselines collapse at long range."
    significance: "It shifts KV compression from input-dependent token selection to a static head-level structure that serving systems can pre-allocate, and the finding that most heads are streaming gives mechanistic support to hybrid full+linear attention architectures."
    limitations: "The paper reports weaker gains on GQA models (a **50%** retrieval ratio vs **25%** for MHA) and requires a per-model gate-training identification phase, and the passkey-calibrated head split's universality across reasoning, code, and multi-hop tasks is not fully settled (reviewer judgment)."
    future_work: "Not stated in the paper; natural extensions are budgeted non-binary head allocations, task-diverse calibration objectives, and retrieval/streaming-aware co-design of GQA/MLA or hybrid architectures (reviewer judgment)."
    resources: "The official implementation is open-sourced at the MIT-HAN Lab GitHub repository (mit-han-lab/duo-attention)."
source: 'manual'
---

## Notes

DuoAttention reframed KV-cache compression: instead of asking which tokens to keep, it asks which heads need long context at all. Its optimization-based identification — gates distilled against the full-cache model's outputs — is a more reliable measurement instrument than attention-score inspection, and the binary retrieval/streaming split gives serving systems a predictable, input-independent memory footprint. The paper is also frequently cited as mechanistic evidence for hybrid designs: if most heads are streaming anyway, replacing them with cheap sliding or linear layers (as in MiniMax-01-style architectures) has principled headroom.

Token 단위 eviction이 아니라 head 단위 구조 분해라는 점에서, KV 압축 연구와 hybrid attention 설계를 잇는 다리 같은 논문이다.
