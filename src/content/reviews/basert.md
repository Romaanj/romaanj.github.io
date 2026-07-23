---
title: "BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators"
arxivId: "2607.19438"
authors: "Fabian Waschkowski, Prabod Rathnayaka, Lukas Wesemann"
lab: "Base Compute, Melbourne, Australia"
date: 2026-07-23
tags: ["on-device", "serving", "kernels"]
topic: 'on-device'
summary: "BaseRT adds hand-written Metal 4 tensor-core kernels to its native Apple-Silicon inference runtime, routing only the compute-bound matrix multiplications of LLM inference (prefill GEMM, MoE expert GEMM, prefill attention) through the M5 generation's per-core Neural Accelerators, reaching up to 6.4x higher prompt-processing throughput than llama.cpp and 3.9x higher than MLX across fifteen model configurations on an Apple M5 Pro."
summary_ko: "BaseRT는 자체 Apple Silicon 네이티브 추론 런타임에 직접 작성한 Metal 4 텐서코어 커널을 추가해, LLM 추론에서 연산-바운드인 부분(프리필 GEMM, MoE 전문가 GEMM, 프리필 어텐션)만 M5 세대의 코어별 Neural Accelerator로 라우팅하며, Apple M5 Pro의 15개 모델 구성에서 llama.cpp 대비 최대 6.4배, MLX 대비 최대 3.9배 높은 프롬프트 처리 처리량을 달성한다."
links: []
resources:
  - label: 'arXiv'
    url: 'https://arxiv.org/abs/2607.19438'
  - label: 'GitHub'
    url: 'https://github.com/basecompute/baseRT'
figures: []
analysis:
  ko:
    background: "온디바이스(로컬) LLM 추론에 대한 수요는 프라이버시·지연시간·연결성·토큰 비용 등 여러 이유로 계속 커지고 있다. Apple Silicon은 통합 메모리 아키텍처(CPU/GPU가 시스템 메모리를 함께 고대역폭으로 공유)와 성숙한 GPU 컴퓨트 스택 덕분에 대표적인 엣지 추론 플랫폼으로 꼽힌다."
    problem: "Apple M5 세대는 GPU 코어마다 전용 Neural Accelerator(온다이 행렬 연산 유닛, Metal 4 텐서 API로 노출)를 새로 도입했지만, 기존 온디바이스 추론 런타임(llama.cpp, MLX)은 이 새 하드웨어 유닛을 아직 충분히 활용하지 못한다."
    prior_limits: "이전 세대 BaseRT를 포함해 기존 런타임들은 SIMD 기반 커널에 의존해왔다 — M5의 전용 행렬 유닛이 주는 이득을 프리필(prompt-processing)의 연산-바운드 구간에서 끌어내지 못한 채 남겨두고 있었다."
    goal: "M5의 Neural Accelerator를 실제로 활용하는 텐서코어 커널을 추가하되, 이미 메모리 대역폭에 의해 병목이 걸린 디코드 구간은 건드리지 않고, 연산-바운드인 프리필 구간에만 정확히 적용하는 것이 목표다."
    method: "밀집(dense) 및 MoE 전문가 GEMM, 융합 전문가 프로젝션, 프리필 어텐션(flash-attention 스타일)을 위한 손수 작성한 Metal 4 협력-텐서 커널 계열을 추가한다. 워크로드-인식 디스패치가 칩 세대를 감지해, M5 계열에서는 연산-바운드 연산(프리필 GEMM·MoE GEMM·프리필 어텐션)에 텐서 경로를, 메모리-바운드인 디코드에는 기존 특화 GEMV/디코드-어텐션 커널을 그대로 쓰며, M5 이전 칩에서는 전체 텐서 경로를 우회해 기존 SIMD 커널과 동일하게 동작한다. 마치 이미 잘 뚫려 있는 좁은 도로(디코드, 대역폭-바운드)는 그대로 두고, 정체가 심한 넓은 교차로(프리필, 연산-바운드)에만 새로 뚫린 고속 우회로를 배정하는 것과 비슷하다."
    key_idea: "핵심은 하드웨어 신규 유닛을 무차별적으로 쓰는 대신, 추론의 두 국면(연산-바운드 프리필 vs. 대역폭-바운드 디코드)을 구분해 이득이 실제로 나는 구간에만 정확히 새 경로를 배정하는 워크로드-인식 디스패치다. 이는 새 하드웨어를 켜켜이 쌓는 대신 병목의 위치를 먼저 진단하고 개입을 국소화하는 접근이다."
    validation: "Apple M5 Pro(통합 메모리 48GB) 한 대에서 Qwen3·Qwen3.5/3.6·Llama 3.2·Gemma 4 계열, 서브-1B부터 35B(4개 MoE 모델 포함)까지 15개 모델 구성으로, llama.cpp(Metal 백엔드)·mlx-lm과 동일 장치·동일 양자화(대부분 4비트, 일부 소형 모델은 8비트)에서 비교한다. 프롬프트 길이 128–2048에서의 프리필 처리량과 128토큰 디코드 처리량(tg128)을 5회 반복 평균으로 측정한다."
    results: "프리필(프롬프트 처리) 처리량은 llama.cpp 대비 최대 6.4배, MLX 대비 최대 3.9배 높으며, 행렬곱이 지배하는 MoE 모델에서 격차가 가장 크다. 디코드 처리량도 llama.cpp 대비 최대 1.75배, MLX 대비 최대 1.33배 앞서지만, 저자들 스스로 이는 대역폭 병목 안에서의 개선이라고 명시한다 — 새 텐서 유닛이 디코드의 근본 상한 자체를 올리지는 못한다."
    comparison: "동일 M5 Pro 장치·동일 체크포인트·동일 양자화 조건에서 llama.cpp(build b9960)와 mlx-lm(0.31.3/MLX 0.32.0)을 기본 설정으로 비교하며, BaseRT(v0.1.6)는 자체 네이티브 가중치 포맷으로 변환한 동일 체크포인트를 사용한다 — 15개 구성 전부에서 llama.cpp 대비 프리필·디코드 모두 앞선다고 보고된다."
    significance: "온디바이스 효율 관점에서, 이 논문은 새 하드웨어 유닛이 있으면 무조건 쓴다가 아니라 추론의 어느 국면이 실제로 그 유닛의 이득을 볼 수 있는가를 먼저 구분하는 것이 성능 개선의 핵심임을 보여준다 — 온디바이스 추론의 성능 상한이 계속 올라가고 있다는 신호이기도 하다."
    limitations: "텐서코어 커널은 연산-바운드 구간에만 적용되며 메모리 대역폭이 정하는 디코드 처리량 상한 자체는 올리지 못한다(논문 명시). 프리필 이득을 실제로 얻으려면 M5 계열 하드웨어와 Metal 4 툴체인이 필요하고, 이전 칩에서는 기존 SIMD 커널로 폴백한다(논문 명시). 평가는 M5 Pro 단일 기기에서만 이루어졌으며, 코어 수·대역폭이 다른 M5 base/Max 등급에서의 동작은 향후 측정 과제로 남아 있다(논문 명시). BaseRT는 이전 연구와 마찬가지로 단일 기기·단일 사용자 추론을 겨냥하며, 연속 배칭·다중 요청 병렬 디코딩·텐서 병렬은 구현하지 않고 Metal 전용이다(논문 명시)."
    future_work: "저자들은 디코드가 여전히 대역폭-바운드임을 인정하면서, 스펙큘레이티브 디코딩이 디코드 작업의 일부를 배치 검증(연산-바운드, 따라서 텐서코어 경로에 적합)으로 바꿔줄 수 있다는 점을 유망한 방향으로 명시한다 — M5 Neural Accelerator와 결합해 디코드 처리량을 간접적으로 끌어올리는 것이 목표다. 또한 프리필 이득이 커진 만큼, 긴 컨텍스트·멀티모달 워크로드 스케줄링에서 프리필-디코드 균형을 재검토할 필요가 있다고 밝힌다(논문 명시)."
    resources: "공개 GitHub 저장소(https://github.com/basecompute/baseRT)가 논문 결론부에 명시되어 있고 접속이 확인되었다."
  en:
    background: "Demand for on-device (local) LLM inference keeps growing for reasons including privacy, latency, connectivity, and token cost. Apple Silicon stands out as an edge-inference platform thanks to its unified memory architecture (CPU and GPU share high-bandwidth access to the same system memory) and a mature GPU compute stack."
    problem: "The Apple M5 generation introduces a dedicated Neural Accelerator per GPU core (on-die matrix units, exposed via the Metal 4 tensor API), but existing on-device inference runtimes (llama.cpp, MLX) do not yet fully exploit this new hardware unit."
    prior_limits: "Prior runtimes, including earlier BaseRT itself, have relied on SIMD-based kernels -- leaving the gains available from M5's dedicated matrix units unrealized specifically in the compute-bound prompt-processing (prefill) phase."
    goal: "Add tensor-core kernels that actually exploit the M5 Neural Accelerators, without touching the already memory-bandwidth-bound decode phase, applying the new path precisely to the compute-bound prefill phase."
    method: "The paper adds a family of hand-written Metal 4 cooperative-tensor kernels for dense and mixture-of-experts GEMM, fused expert projections, and flash-attention-style prefill attention. Workload-aware dispatch detects the chip generation: on M5-family hardware, the tensor path handles compute-bound operators (prefill GEMM, MoE expert GEMM, prefill attention) while memory-bound decode continues to use BaseRT's existing specialized GEMV and decode-attention kernels; on pre-M5 chips, the entire tensor path is bypassed and behavior is identical to the SIMD kernels. It is a bit like leaving an already-efficient narrow road (decode, bandwidth-bound) untouched, while routing only the congested wide intersection (prefill, compute-bound) through a newly built bypass."
    key_idea: "The core idea is not to apply the new hardware unit indiscriminately, but to first distinguish the two phases of inference (compute-bound prefill vs. bandwidth-bound decode) via workload-aware dispatch, and route the new path exactly where it helps. This is diagnosing where the bottleneck actually sits before localizing the intervention, rather than layering new hardware support everywhere."
    validation: "Benchmarked on a single Apple M5 Pro (48GB unified memory) across fifteen model configurations spanning Qwen3, Qwen3.5/3.6, Llama 3.2, and Gemma 4, from sub-1B to 35B parameters (including four MoE models), against llama.cpp (Metal backend) and mlx-lm on the same device and matched quantization (mostly 4-bit, 8-bit for some smaller models). Prefill throughput is measured at prompt lengths 128-2048 and decode throughput over 128 generated tokens, averaged over five repetitions."
    results: "Prefill (prompt-processing) throughput reaches up to 6.4x higher than llama.cpp and 3.9x higher than MLX, with the largest margins on matmul-dominated MoE models. Decode throughput also leads by up to 1.75x over llama.cpp and 1.33x over MLX, but the authors themselves state this is an improvement within the bandwidth-bound ceiling -- the new tensor units do not raise decode's fundamental throughput ceiling."
    comparison: "Compared against llama.cpp (build b9960) and mlx-lm (0.31.3/MLX 0.32.0) with default settings, on the same M5 Pro device, same checkpoints, and matched quantization, with BaseRT (v0.1.6) loading the same checkpoints converted to its native weight format -- BaseRT is reported faster than llama.cpp on both prefill and decode across all fifteen configurations."
    significance: "From an on-device-efficiency standpoint, this paper shows that the key to performance gains from new hardware units is not use-it-wherever-available but first identifying which phase of inference can actually benefit from that unit -- also a signal that the performance ceiling for on-device inference keeps rising."
    limitations: "The tensor-core kernels apply only to compute-bound phases and do not raise the decode throughput ceiling set by memory bandwidth (stated in the paper). Realizing the prefill gains requires M5-family hardware and a Metal 4 toolchain; earlier chips fall back to the existing SIMD kernels (stated in the paper). Evaluation is conducted on a single M5 Pro device only, with behavior on the M5 base and Max tiers (which differ in core count and bandwidth) left for future measurement (stated in the paper). Like prior work, BaseRT targets single-device, single-user inference and does not implement continuous batching, multi-request parallel decoding, or tensor parallelism, and targets Metal exclusively (stated in the paper)."
    future_work: "The authors state that while decode remains memory-bound, speculative decoding could turn part of the decode workload into a batched verification step whose matmuls are compute-bound and therefore amenable to the tensor-core path -- combining it with the M5 Neural Accelerators is named as a promising direction for raising decode throughput indirectly. They also state that the prefill advantage invites revisiting the prefill-decode balance in scheduling long-context and multimodal workloads (stated in the paper)."
    resources: "A public GitHub repository (https://github.com/basecompute/baseRT) is stated in the paper's conclusion and was confirmed reachable."
thread:
  ko: |-
    온디바이스 LLM 추론 런타임 계열(llama.cpp, MLX, 그리고 BaseRT 자신의 이전 버전)은 지금까지 대체로 SIMD 기반 커널·통합 메모리 활용·양자화 지원을 축으로 발전해왔다 — Apple Silicon의 GPU 컴퓨트 스택을 얼마나 잘 짜느냐의 싸움이었다.

    이 논문의 전환점은 하드웨어 자체의 변화다: M5 세대가 코어마다 전용 행렬 연산 유닛을 새로 얹으면서, 기존 SIMD 중심 런타임들이 활용하지 못하는 새로운 자원이 생겼다. 이 논문이 하는 일은 단순히 그 유닛을 쓰는 커널을 추가하는 것이 아니라, 추론의 어느 국면(프리필 vs. 디코드)이 그 유닛으로부터 실제 이득을 볼 수 있는지 구분해 정확히 그곳에만 배정하는 워크로드-인식 디스패치를 설계한 것이다.

    이 흐름이 여는 질문은, 디코드의 대역폭 병목 자체를 우회할 방법이다. 저자들이 직접 언급하듯 스펙큘레이티브 디코딩은 디코드의 일부를 연산-바운드 배치 검증으로 바꿔 텐서 유닛의 이득을 디코드까지 끌어올 수 있는 유망한 경로다 — 이는 온디바이스 추론 런타임이 다음으로 마주할 자연스러운 다음 단계로 보인다.
  en: |-
    On-device LLM inference runtimes (llama.cpp, MLX, and BaseRT's own earlier versions) have mostly advanced along an axis of SIMD-based kernels, unified-memory exploitation, and quantization support -- a competition over how tightly you can engineer against Apple Silicon's GPU compute stack.

    This paper's turning point is a hardware change: the M5 generation adds a dedicated matrix unit per core, creating a resource that existing SIMD-centric runtimes cannot exploit. What this paper does is not simply add a kernel that uses that unit, but design workload-aware dispatch that first distinguishes which phase of inference (prefill vs. decode) can actually benefit from it, and routes only that phase through the new path.

    The question this opens is how to route around decode's bandwidth bottleneck itself. As the authors note, speculative decoding could turn part of decode into a compute-bound batched verification step, pulling the tensor unit's benefit into decode as well -- a natural next step for on-device inference runtimes to pursue.
sparks:
  - ko: "저자들이 직접 명시한 미래 과제대로, 스펙큘레이티브 디코딩과 M5 Neural Accelerator를 결합해 디코드의 배치 검증 부분을 텐서코어 경로로 옮기면 대역폭 병목을 얼마나 우회할 수 있는지 실측해보는 것은 자연스러운 다음 실험이다."
    en: "As the authors themselves flag as future work, actually measuring how much combining speculative decoding with the M5 Neural Accelerators (routing decode's batched-verification step through the tensor-core path) can work around the bandwidth bottleneck is a natural next experiment."
  - ko: "평가가 M5 Pro 한 기종에만 국한되어 있다고 저자들이 명시한다 — 코어 수·메모리 대역폭이 다른 M5 base·Max 등급에서 같은 워크로드-인식 디스패치가 얼마나 잘 일반화되는지 측정하면 이 접근의 하드웨어 의존성을 더 분명히 할 수 있다."
    en: "The authors state evaluation is limited to a single M5 Pro unit -- measuring how well the same workload-aware dispatch generalizes to the M5 base and Max tiers, which differ in core count and memory bandwidth, would clarify how hardware-dependent this approach is."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
