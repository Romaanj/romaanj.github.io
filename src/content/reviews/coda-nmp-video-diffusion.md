---
title: "CODA: Algorithm-Hardware Co-design for Edge Video Diffusion via NMP-Enabled Compute-Cache Operator Disaggregation"
arxivId: "2607.14908"
date: 2026-07-18
tags: ["on-device", "video-diffusion", "hardware-codesign"]
topic: 'on-device'
summary: "Moves video-diffusion's cross-timestep activation cache off the GPU onto a DIMM-side near-memory-processing engine so caching stops competing with compute for VRAM and bandwidth on edge GPUs."
summary_ko: '비디오 디퓨전의 시간축 캐시를 GPU에서 DIMM 옆의 근접 메모리 처리(NMP) 엔진으로 옮겨, 엣지 GPU에서 캐싱이 연산과 VRAM·대역폭을 두고 경쟁하지 않게 만드는 알고리즘-하드웨어 공동설계.'
links: ["sangam", "mobilewan"]
resources:
  - label: "arXiv abstract"
    url: "https://arxiv.org/abs/2607.14908"
figures:
  - src: /figures/coda-nmp-video-diffusion/fig1.png
    caption: "(a) On an RTX 4090 edge GPU, offloading the overflowed cache to host DRAM erases the compute savings caching was supposed to deliver; (b) limited VRAM forces the cache path onto PCIe, starving GPU utilization; (c) CODA instead disaggregates the cache operators onto a lightweight DIMM-side NMP engine and overlaps it with xPU compute."
    caption_ko: "(a) RTX 4090 엣지 GPU에서 넘친 캐시를 호스트 DRAM으로 내리면 캐싱이 주려던 연산 절감 효과 자체가 사라진다. (b) VRAM이 부족하면 캐시 경로가 PCIe에 묶여 GPU 활용률이 떨어진다. (c) CODA는 대신 캐시 연산자를 가벼운 DIMM 측 NMP 엔진으로 분리해 xPU 연산과 겹쳐 실행한다."
    credit: "Figure 1 from arXiv:2607.14908 — authors' figure"
analysis:
  ko:
    background: '비디오 디퓨전 모델은 디노이징 스텝 사이에서 중간 활성값(피처)을 재사용하는 시간축(cross-timestep) 캐싱으로 중복 연산을 줄이는 것이 표준 관행이 되었다. 하지만 엣지 GPU에서는 이 캐시 자체가 VRAM을 크게 잡아먹어, 연산은 줄어도 메모리가 새로운 병목이 된다.'
    problem: '엣지 배포(프라이버시를 지키는 로컬 비디오 생성이 대표적 동기)에서는 VRAM이 넘치면 캐시를 호스트 DRAM으로 내려야 하는데, 이러면 캐시 경로가 PCIe 대역폭에 묶여 GPU 활용률이 떨어지고 캐싱이 주려던 이득 자체가 사라진다.'
    prior_limits: '기존의 캐시 기반 가속 기법들은 캐시를 어디에 둘지, 그 캐시 자체가 새로운 병목이 될 수 있다는 점을 소프트웨어 알고리즘 층위에서만 다뤄왔고, 캐시 경로를 압축(연산)에서 분리된 별도의 메모리 바운드 작업으로 취급하지 않았다.'
    goal: '엣지 GPU에서 시간축 캐싱의 압축 이득을 유지하면서, 캐시 자체가 만드는 VRAM·PCIe 병목을 제거하는 것이 목표다 — 생성 품질은 그대로 유지한 채로.'
    method: 'CODA는 워크로드를 두 하드웨어 경로로 분리한다: 조밀한 트랜스포머 연산은 그대로 xPU(주 연산 다이)에 두고, 메모리 바운드인 시간축 캐시 경로(읽기·쓰기·재사용)는 DIMM 옆의 근접 메모리 처리(NMP) 엔진으로 내린다. 분류기 자유 안내(CFG)가 원래 구조적으로 독립적인 조건부·무조건부 두 순전파를 갖는다는 점을 활용해, 연산 경로와 캐시 경로의 실행을 겹쳐(overlap) 처리한다.'
    key_idea: '핵심 통찰은 "캐싱이 연산을 줄여준다"는 가정과 "그 캐시를 어디서 처리하느냐"를 분리한 것이다. 비유하자면, 요리(연산)와 냉장고 정리(캐시 관리)를 같은 사람이 순서대로 하는 대신, 냉장고 정리를 옆에서 동시에 처리해주는 보조를 두는 것과 같다.'
    validation: 'RTX 4090(24GB GDDR6X) + Intel i9-13900K + 128GB DRAM(PCIe 4.0 x16) 구성의 실제 엣지 하드웨어에서 여러 비디오 디퓨전 모델에 대해 종단간(end-to-end) 지연시간과 에너지 효율을 측정했다.'
    results: 'SOTA 캐싱 베이스라인 대비 종단간 속도 향상 최대 **1.80배**, 에너지 효율 최대 **1.74배**를 보고하며, 생성 품질은 그대로 유지된다.'
    comparison: '비교 대상은 캐시를 호스트 DRAM으로 오프로드하는 기존 SOTA 캐싱 방식이며, 논문은 이 방식이 엣지 환경에서 PCIe 대역폭에 묶여 성능을 잃는 지점을 정량적으로 보여준다. CODA는 순수 소프트웨어 캐시 압축(예: 캐시 자체를 양자화하거나 축소)이 아니라 하드웨어 배치를 바꾸는 접근이라는 점에서 다른 실험군이다.'
    significance: '이 논문은 "캐싱이 연산을 아껴주지만 병목을 다른 곳(메모리)으로 옮긴다"는, 언어모델 KV 캐시 압축 연구에서도 반복적으로 관찰되는 패턴이 비디오 디퓨전이라는 전혀 다른 도메인에서, 그것도 하드웨어 배치 층위에서 독립적으로 재확인된 사례다.'
    limitations: '논문 내 명시 없음 (본 리뷰는 초록과 Figure 1 캡션을 근거로 작성됨) (리뷰어 판단). 다만 평가가 단일 GPU급(RTX 4090) 엣지 구성에 한정되어 있어, 더 작은 모바일/임베디드 급이나 다중 GPU 서버급으로의 일반화는 논문에서 다뤄지지 않는다 (리뷰어 판단).'
    future_work: '논문 내 명시 없음.'
    resources: 'arXiv 프리프린트가 유일하게 확인된 공개 자료다 (리뷰어 판단 — 코드·하드웨어 설계 공개 여부는 확인되지 않음).'
  en:
    background: 'Cross-timestep caching — reusing intermediate activations/features across denoising steps — has become the standard way to cut redundant compute in video diffusion. On edge GPUs, though, the cache itself consumes large amounts of VRAM, so memory becomes the new bottleneck even as compute drops.'
    problem: 'In edge deployment (privacy-preserving local video generation is the paper''s stated motivating use case), when VRAM overflows the cache must spill to host DRAM — which ties the cache path to PCIe bandwidth, starves GPU utilization, and erases the very gains caching was supposed to deliver.'
    prior_limits: 'Prior cache-based acceleration methods treat "where to place the cache" and "the cache becoming a new bottleneck" purely as a software/algorithm-layer concern — none disaggregate the cache path as a separate, memory-bound workload distinct from compute.'
    goal: 'Keep the compute savings of cross-timestep caching on edge GPUs while removing the VRAM/PCIe bottleneck the cache itself creates, without sacrificing generation quality.'
    method: 'CODA splits the workload across two hardware paths: dense Transformer compute stays on the main xPU compute die, while the memory-bound cross-timestep cache path (read/write/reuse) is offloaded to a DIMM-side near-memory-processing (NMP) engine. It exploits classifier-free guidance (CFG)''s two structurally independent forward passes (conditional/unconditional) as a natural point to overlap compute-path and cache-path execution.'
    key_idea: 'The core insight is decoupling "caching reduces compute" from "where that cache gets processed." An analogy: instead of one person doing the cooking (compute) and the fridge-organizing (cache management) in sequence, give the fridge-organizing to an assistant who works alongside, concurrently.'
    validation: 'Measured end-to-end latency and energy efficiency across multiple video-diffusion models on real edge hardware: an RTX 4090 (24GB GDDR6X) with an Intel i9-13900K and 128GB DRAM over PCIe 4.0 x16.'
    results: 'Reports up to **1.80×** end-to-end speedup and **1.74×** higher energy efficiency versus SOTA caching baselines, at matched generation quality.'
    comparison: 'The comparison point is existing SOTA caching that spills overflow to host DRAM; the paper quantifies exactly where that approach becomes PCIe-bound on edge hardware. CODA is a hardware-placement change rather than a pure-software cache-compression method (e.g., quantizing or shrinking the cache itself), making it a different kind of intervention than software-only baselines.'
    significance: 'This independently reconfirms — in a completely different domain (video diffusion) and at the hardware-placement layer — a pattern repeatedly observed in language-model KV-cache compression research: caching saves compute but shifts the bottleneck elsewhere (here, to memory/PCIe bandwidth).'
    limitations: 'Not stated in the paper (this review is based on the abstract and the Figure 1 caption) (reviewer judgment). Evaluation is limited to a single edge-GPU-class configuration (RTX 4090); generalization to smaller mobile/embedded targets or multi-GPU server settings is not addressed by the paper (reviewer judgment).'
    future_work: 'Not stated in the paper.'
    resources: 'The arXiv preprint is the only verified public resource (reviewer judgment — whether code or the hardware design is separately released is not confirmed).'
thread:
  ko: |-
    비디오 디퓨전 가속 연구는 최근 몇 년간 "어떻게 더 똑똑하게 캐싱할까"(어느 스텝을, 어느 피처를 재사용할지)에 집중해 왔다. CODA는 이 흐름에 다른 축 하나를 더한다 — 캐싱 알고리즘 자체는 그대로 두고, 그 캐시를 "어느 하드웨어가 처리하느냐"를 바꾸는 것이다.

    개념적 전환은 캐시 경로를 압축(compute) 문제가 아니라 독립된 메모리 바운드 워크로드로 인정하고, 이를 위한 전용 하드웨어(DIMM 측 NMP)를 배치하는 것이다. 이는 소프트웨어 최적화만으로는 도달할 수 없는 영역 — 캐시가 VRAM·PCIe와 경쟁하는 물리적 병목 — 을 정면으로 다룬다.

    이 논문이 여는 질문은, 같은 "캐시가 새로운 병목이 된다"는 패턴이 언어모델의 KV 캐시 서빙에서도 하드웨어 재배치로 해소될 수 있는지, 그리고 하드웨어 분리(NMP)와 소프트웨어 압축(양자화·축출)이 서로 경쟁하는 대안인지 아니면 결합 가능한 보완책인지이다.
  en: |-
    Video-diffusion acceleration research has, for the past few years, focused on "how to cache more cleverly" — which steps, which features to reuse. CODA adds a different axis to that line: leave the caching algorithm itself unchanged, and instead change *which hardware* processes that cache.

    The conceptual shift is recognizing the cache path as an independent, memory-bound workload rather than a compute problem, and giving it dedicated hardware (DIMM-side NMP) to match. This addresses a bottleneck — the cache competing with compute for VRAM and PCIe bandwidth — that software-only optimization can't reach.

    The question this opens is whether the same "caching becomes the new bottleneck" pattern seen in LLM KV-cache serving could likewise be resolved by hardware re-placement, and whether hardware disaggregation (NMP) and software compression (quantization, eviction) are competing alternatives or composable complements.
sparks:
  - ko: "CFG의 두 독립적 순전파를 겹쳐 실행하는 트릭이 CFG를 쓰지 않는(또는 distillation으로 CFG를 제거한) 비디오 디퓨전 모델에서는 어떤 대체 오버랩 지점을 찾아야 하는가?"
    en: "If a video-diffusion model doesn't use CFG (or has it distilled away), what alternative overlap point would this compute/cache-path disaggregation need to find?"
  - ko: "DIMM 측 NMP 엔진이 모바일·임베디드급 하드웨어(더 제한된 메모리 계층)에도 이식 가능한지는 이 논문의 RTX 4090급 평가만으로는 알 수 없다."
    en: "Whether the DIMM-side NMP engine is portable to smaller mobile/embedded hardware with a more constrained memory hierarchy is not answerable from this paper's RTX-4090-class evaluation alone."
  - ko: "하드웨어 분리(NMP)와 캐시 자체의 소프트웨어 압축(양자화·축출)을 함께 적용하면 이득이 더해지는지, 아니면 하나가 다른 하나의 필요성을 없애는지는 열린 질문이다."
    en: "Whether combining hardware disaggregation (NMP) with software-side cache compression (quantization, eviction) is additive, or whether one makes the other unnecessary, is an open question."
source: "autosweep"
---

## Notes
