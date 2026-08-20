---
title: "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution"
arxivId: "2608.16157"
date: 2026-08-21
tags: ["serving", "kv-cache", "on-device", "moe"]
topic: 'serving'
summary: "An edge-native MoE serving system that continuously repartitions GPU memory between KV cache and resident experts at runtime, letting consumer GPUs serve up to 753B-parameter models with 1.3-2.3x higher decode throughput than prior edge engines."
summary_ko: '엣지 GPU에서 KV 캐시와 상주 전문가(expert) 슬롯 사이의 GPU 메모리 배분을 실행 중에 계속 재조정하는 MoE 서빙 시스템으로, 소비자용 GPU 한 대로 최대 7530억 파라미터 모델까지 서빙하며 기존 엣지 엔진 대비 1.3~2.3배 높은 디코드 처리량을 낸다.'
links: ["kvquant", "kivi", "mobilellm", "flashdrive-vla-inference"]
resources: [{"label": "arXiv abstract", "url": "https://arxiv.org/abs/2608.16157"}, {"label": "arXiv PDF", "url": "https://arxiv.org/pdf/2608.16157"}, {"label": "Project page (flashml.ai)", "url": "https://flashml.ai"}]
figures:
  - src: /figures/freetoken-edge-moe-serving/fig1.svg
    caption: "FreeToken's serving-stack overview: runtime GPU-memory repartitioning between KV cache and resident expert slots, plus bandwidth-adaptive CPU/GPU execution."
    caption_ko: "FreeToken 서빙 스택 개요: KV 캐시와 상주 전문가 슬롯 사이의 실행 중 GPU 메모리 재배분, 그리고 대역폭 적응형 CPU/GPU 실행."
    credit: "Figure from arXiv:2608.16157 — authors' figure"
  - src: /figures/freetoken-edge-moe-serving/fig2.svg
    caption: "Main throughput comparison against edge serving baselines (llama.cpp, Ollama, KTransformers, MoE-Infinity) on an RTX 5090."
    caption_ko: "RTX 5090에서 엣지 서빙 베이스라인(llama.cpp, Ollama, KTransformers, MoE-Infinity) 대비 주요 처리량 비교."
    credit: "Figure from arXiv:2608.16157 — authors' figure"
analysis:
  ko:
    background: '개방형 가중치(open-weight) 프론티어 모델은 빠르게 늘고 있지만, 이를 서빙하는 소프트웨어는 여전히 데이터센터급 인프라를 전제로 설계된다. 반면 개인이 실제로 가진 하드웨어는 8GB 노트북 GPU부터 워크스테이션급 GPU까지 매우 이질적이며, CPU-GPU 대역폭 균형도 기기마다 다르다.'
    problem: '코딩 에이전트나 툴 사용 에이전트 같은 실제 워크로드는 세션이 진행되며 실행 패턴이 계속 바뀐다. 특히 멀티턴 세션에서는 턴이 쌓일수록 KV 캐시 수요는 커지는 반면 전문가(expert) 작업셋 크기는 거의 고정돼 있어, 첫 턴에 정한 자원 배분이 몇 턴 뒤에는 이미 최적이 아니게 된다.'
    prior_limits: 'llama.cpp, Ollama, KTransformers, MoE-Infinity 같은 기존 엣지 서빙 엔진들은 어떤 전문가를 GPU/CPU 중 어디에 둘지, GPU 메모리를 KV 캐시와 전문가 가중치 사이에 어떻게 나눌지를 구동 시점에 한 번만 정하고 고정한다. 재조정하려면 엔진을 재시작해야 하는데, 이는 그 자체로 비용이 크다. 특히 MoE-Infinity는 요청 간 KV 캐시를 아예 유지하지 않아 멀티턴에서 처리량이 급락한다(첫 턴 대비 두 번째 턴에서 31% 손실).'
    goal: '재시작 없이, 세션이 진행되는 동안에도 GPU/CPU/PCIe 자원을 KV 캐시와 전문가 상주 슬롯 사이에서 실시간으로 재매핑할 수 있는 엣지 네이티브 MoE 서빙 시스템을 만드는 것이 목표다.'
    method: '핵심은 세 가지다. (1) 런타임 KV/전문가 메모리 재분할: 스케줄러의 안전 지점(safe point)마다 GPU 메모리를 페이지 단위 KV 캐시(SGLang/vLLM 방식의 radix-tree 접두사 재사용)와 완전한 전문가 슬롯 사이에서 재구성한다. (2) 대역폭 적응형 실행: CPU에 상주하는 전문가는 아키텍처별 SIMD와 커널 내부 역양자화(in-kernel dequantization)로 읽어, CPU 경로를 순수하게 대역폭 제한(bandwidth-bound) 상태로 유지한다. 이중 버퍼링된 PCIe 5.0 x16 전송(측정된 상한 52.7GB/s)으로 전문가 가중치 가져오기를 GPU 연산 뒤에 완전히 숨긴다 — 이 이중 버퍼를 끄면 4k/8k/16k 토큰 프롬프트에서 각각 처리량의 19%/25%/26%를 잃는다. (3) 시맨틱 앵커(semantic anchor): 에이전트가 툴 호출로 히스토리를 편집하면, 완전 어텐션(full-attention) 레이어는 편집 지점까지의 KV 캐시를 재사용하고, 순환(recurrent) 레이어는 그 지점에 고정된 체크포인트에서 재개해 전체 재-프리필을 피한다.'
    key_idea: '비유하자면, 기존 엔진들이 이사 갈 때 방 배치를 한 번 정하고 다시는 바꾸지 않는 것과 같다면, FreeToken은 방문객(에이전트 세션)이 늘어나거나 짐(KV 캐시)이 쌓일 때마다 가구 배치를 실시간으로 다시 짜는 셈이다. GPU 메모리를 KV 캐시와 전문가 슬롯이라는 두 자원 사이의 재조정 가능한 예산으로 다루는 것이 핵심 발상이다.'
    validation: 'Qwen3.6은 모든 엔진에서 동일하게 BF16으로, DeepSeek-V4-Flash는 모든 엔진에서 동일하게 네이티브 MXFP4로 맞춰 가중치를 비트 단위로 동일하게 유지한 채 엔진 간 비교를 했다 — 양자화 방식 차이가 결과에 섞이지 않도록 통제한 것이다. 이중 버퍼 온/오프 절제 실험(ablation)으로 PCIe 전송 은닉 메커니즘의 기여를 분리해 측정했고, 8GB 노트북부터 워크스테이션 GPU까지 5개의 서로 다른 소비자 하드웨어 등급에서 검증했다.'
    results: 'RTX 5090에서 Qwen3.6-35B-A3B는 77-83 tok/s, DeepSeek-V4-Flash-284B는 22-25 tok/s로, 가장 강력한 엣지 베이스라인 대비 1.5-2.3배 높은 디코드 처리량을 냈고, 이 격차는 에이전트 워크로드가 멀티턴으로 진행될수록 더 벌어졌다. 8GB RTX 4060 노트북에서는 35B 모델을 39.3 tok/s로 서빙해, 실제 프로덕션에서 보고된 Codex의 중앙값 디코드 속도(33 tok/s)를 넘어섰다. 단일 RTX PRO 6000 워크스테이션 GPU에서는 7530억 파라미터 GLM-5.2를 14.9 tok/s로 서빙해 llama.cpp(7.3 tok/s)의 2.0배였다(전문가 가중치는 두 엔진에서 비트 단위로 동일).'
    comparison: 'MoE-Infinity는 첫 워크로드(W1)만 서빙 가능하고(8.8 tok/s) 더 긴 프롬프트 워크로드는 전문가별 프리필 스테이징 상한 때문에 아예 실패한다. Ollama와 MoE-Infinity는 DeepSeek-V4를 지원하지 않아 비교 대상에서 제외됐다. FreeToken은 여섯 개의 엔진×워크로드 조합 중 다섯에서 가장 낮은 첫 토큰 지연시간(TTFT)을 기록했다.'
    significance: '이 논문 자체는 자기회귀(AR) MoE 모델 서빙에 관한 것으로 dLLM이나 KV 비트 배분(water-filling) 주장과는 직접 겹치지 않는다. 하지만 GPU 메모리를 KV 캐시와 또 다른 자원 클래스(여기서는 전문가 슬롯) 사이의 실행 중 재조정 가능한 예산으로 다루는 패턴과, 이중 버퍼 온/오프라는 깔끔한 대역폭-루프라인(roofline) 절제 실험 방법론은 서빙 시스템 일반에 참고할 만하다.'
    limitations: '평가된 모델은 모두 자기회귀 MoE LLM(Qwen3.6, DeepSeek-V4-Flash, GLM-5.2)이며, 확산 언어모델(diffusion LM) 워크로드는 테스트되지 않았다. 논문은 대역폭 적응형 전문가 상주 방식이나 시맨틱 앵커 접두사 재사용 방식이 단일한 인과적(causal) KV 성장 방향이 없는 마스크/블록 확산 디코딩 루프에 맞게 재작업이 필요한지 논의하지 않는다(리뷰어 판단). 또한 범위 전체가 단일 머신·단일 GPU에 한정되어 있어 멀티 GPU 텐서 병렬이나 분산 서빙 환경에 대한 주장은 없다.'
    future_work: '논문 내 명시된 향후 연구 방향은 없다(논문 내 명시 없음).'
    resources: '시스템은 flashml.ai에서 공개된다고 논문에 명시되어 있다.'
  en:
    background: 'Open-weight frontier models are increasingly available, but the software that serves them still largely assumes datacenter-scale infrastructure. Meanwhile the hardware people actually own is highly heterogeneous, from an 8GB laptop GPU to a workstation-class GPU, and the CPU-GPU bandwidth balance differs from machine to machine.'
    problem: 'Real workloads such as coding agents and tool-using agents keep changing their execution pattern as a session progresses. In multi-turn sessions in particular, KV-cache demand grows with each turn while the expert working set stays roughly fixed in size, so a resource split chosen on turn one is no longer optimal a few turns later.'
    prior_limits: 'Existing edge serving engines such as llama.cpp, Ollama, KTransformers, and MoE-Infinity decide once, at launch, which experts sit on GPU vs. CPU and how GPU memory splits between the KV cache and expert weights, then keep that split fixed. Re-tuning it requires restarting the engine, which is itself expensive. MoE-Infinity in particular keeps no KV cache across requests at all, so its throughput collapses in multi-turn use, a 31% drop from the first turn to the second.'
    goal: 'The goal is an edge-native MoE serving system that can remap GPU/CPU/PCIe resources between the KV cache and resident expert slots in real time, as a session runs, without restarting the engine.'
    method: 'The core has three parts. (1) Runtime KV/expert memory repartitioning: at each scheduler safe point, GPU memory is rebuilt between paged KV cache (radix-tree prefix reuse, following SGLang/vLLM) and complete expert slots. (2) Bandwidth-adaptive execution: CPU-resident experts are read with architecture-specific SIMD plus in-kernel dequantization, keeping the CPU path strictly bandwidth-bound; a double-buffered PCIe 5.0 x16 transfer (a measured ceiling of 52.7 GB/s) hides expert-weight fetch fully behind GPU compute, and disabling the second buffer costs 19%/25%/26% of throughput at 4k/8k/16k-token prompts. (3) Semantic anchors: when an agentic tool call edits history, full-attention layers reuse their KV cache up to the edit boundary while recurrent layers resume from a checkpoint anchored at that same boundary, avoiding a full re-prefill.'
    key_idea: 'If existing engines are like deciding a room layout once when you move in and never rearranging it, FreeToken re-arranges the furniture in real time as visitors, meaning agent sessions, arrive and belongings, meaning the KV cache, pile up. The core idea is to treat GPU memory as a continuously re-negotiable budget split between two resources, the KV cache and the expert slots, rather than a fixed allocation.'
    validation: 'Cross-engine comparisons hold weights bit-identical, Qwen3.6 in BF16 and DeepSeek-V4-Flash in its native MXFP4 across every engine, so no quantization-scheme difference confounds the result. A double-buffer on/off ablation isolates the contribution of the PCIe-transfer-hiding mechanism specifically, and five distinct consumer hardware tiers were tested, from an 8GB laptop to a workstation GPU.'
    results: 'On an RTX 5090, Qwen3.6-35B-A3B reaches 77-83 tok/s and DeepSeek-V4-Flash-284B reaches 22-25 tok/s, 1.5-2.3x the strongest edge baseline, with the margin widening further under increasingly agentic, multi-turn workloads. On an 8GB RTX 4060 laptop it serves a 35B model at 39.3 tok/s, exceeding the reported 33 tok/s median decode speed of Codex in production. On a single RTX PRO 6000 workstation GPU it serves the 753B-parameter GLM-5.2 at 14.9 tok/s, twice the 7.3 tok/s reached by llama.cpp, with bit-identical expert weights across both engines.'
    comparison: 'MoE-Infinity serves only the first workload tier (8.8 tok/s) and fails longer-prompt workloads outright because of its per-expert prefill-staging cap; Ollama and MoE-Infinity both lack DeepSeek-V4 support and are excluded from that comparison. FreeToken posts the lowest time-to-first-token in five of six engine-by-workload cells tested.'
    significance: 'The paper itself is about serving autoregressive MoE models and makes no dLLM or KV bit-allocation (water-filling) claim, so there is no direct overlap there. What does transfer is the pattern of treating GPU memory as a runtime-renegotiable budget split between the KV cache and a second resource class, here the expert slots, plus the clean double-buffer-ablation methodology for isolating the contribution of a bandwidth-hiding mechanism, both useful reference points for serving-systems work generally.'
    limitations: 'Every evaluated model is an autoregressive MoE LLM, Qwen3.6, DeepSeek-V4-Flash, GLM-5.2, and no diffusion-LM workload is tested. The paper does not discuss whether the bandwidth-adaptive expert-residency scheme or the semantic-anchor prefix reuse would need rework for a masked or block-diffusion decoding loop, which lacks a single causal direction of KV growth to anchor on (reviewer judgment). The scope is also single-machine, single-GPU throughout, so no claim is made about multi-GPU tensor-parallel or disaggregated serving.'
    future_work: 'Not stated in the paper.'
    resources: 'The paper states the system is released at flashml.ai.'
thread:
  ko: |-
    엣지·소비자 GPU에서 대형 모델을 서빙하려는 시도는 llama.cpp의 레이어 단위 오프로딩, FlexGen과 DeepSpeed-Inference의 처리량 지향 스트리밍, PowerInfer의 활성화 통계 기반 뉴런 분할처럼 계속 이어져 왔다. 이 계보가 공유하는 전제는 "GPU와 CPU 사이에서 무엇을 어디에 둘지는 시작할 때 한 번 정하면 된다"는 것이었고, KTransformers 같은 최신 엔진도 라우팅된 전문가를 CPU에 고정해 두는 방식으로 이 전제를 이어받았다.

    FreeToken이 바꾸는 지점은 이 배치 결정을 일회성 설계 선택에서 실행 중 재협상 가능한 자원으로 옮긴 것이다. 에이전트 워크로드가 세션 내내 형태를 바꾼다는 관찰, 즉 KV 캐시는 턴이 쌓일수록 자라고 전문가 작업셋은 거의 고정이라는 관찰에서 출발해, GPU 메모리 분할과 CPU/GPU 실행 경로 자체를 스케줄러 안전 지점마다 다시 짤 수 있는 대상으로 만들었다. SGLang과 vLLM의 페이지형 KV 캐시와 radix 접두사 재사용을 그대로 가져다 쓰면서, 거기에 무엇을 CPU에 상주시킬지까지 같은 수준의 런타임 유연성으로 끌어올린 셈이다.

    이 작업이 여는 질문은 두 방향이다. 하나는 이 재조정 가능한 메모리-예산 프레임을 하이브리드 아키텍처(전체 어텐션과 순환/SSM 레이어 혼합)로 얼마나 더 밀어붙일 수 있는가이고, 다른 하나는 애초에 인과적 KV 성장 방향이 없는 디코딩 방식(마스크·블록 확산 등)에서는 시맨틱 앵커에 해당하는 개념이 무엇이 되어야 하는가이다.
  en: |-
    Serving large models on edge and consumer GPUs is a line of work that runs through llama.cpp layer-wise offloading, FlexGen and DeepSpeed-Inference throughput-oriented weight streaming, and PowerInfer activation-statistics-based neuron splitting. What this lineage shared was the assumption that deciding what lives on GPU versus CPU is a one-time choice made at launch, and even recent engines like KTransformers inherit that assumption by pinning routed experts to CPU for the whole run.

    FreeToken shifts that placement decision from a one-time design choice to a continuously re-negotiable runtime resource. Starting from the observation that agentic workloads keep changing shape across a session, KV-cache demand growing with each turn while the expert working set stays roughly fixed, it makes both the GPU memory split and the CPU/GPU execution path themselves reconfigurable at every scheduler safe point. It borrows the paged KV cache and radix prefix reuse of SGLang and vLLM wholesale, then lifts the question of what resides on CPU to that same level of runtime flexibility.

    This opens two directions. One is how far this re-negotiable memory-budget framing can be pushed into hybrid architectures that mix full attention with recurrent or SSM layers. The other is what the semantic-anchor concept should even mean for decoding schemes, masked or block diffusion among them, that have no single causal direction of KV growth to anchor on in the first place.
sparks:
  - ko: "논문은 순수 자기회귀 MoE 모델만 다룬다 — 같은 런타임 KV/전문가 메모리 재분할 아이디어를 전체 어텐션과 순환 레이어가 섞인 하이브리드 아키텍처로 확장하면 안전 지점을 어떻게 다시 정의해야 할까?"
    en: "The paper covers only pure autoregressive MoE models — extending the same runtime KV/expert memory-repartitioning idea to hybrid architectures mixing full attention with recurrent layers, how would scheduler safe points need to be redefined?"
  - ko: "시맨틱 앵커는 에이전트의 툴 호출이 히스토리를 편집하는 상황을 전제로 설계됐다 — 히스토리 편집이 아니라 모델 자신의 디코딩 결정(예: 재마스킹)이 캐시를 무효화하는 경우에도 같은 체크포인트-재개 아이디어가 성립할까?"
    en: "Semantic anchors are designed around an agent's tool call editing history — would the same checkpoint-and-resume idea hold when it is the model's own decoding decisions, rather than history edits, that invalidate part of the cache?"
source: "autosweep"
---

## Notes
