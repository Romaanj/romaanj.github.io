---
title: "HyMCache: A KV Cache Framework for Multi-Turn LLM Serving with CXL-Hybrid Memory"
arxivId: "2607.18141"
authors: "Hakbeom Jang, Inho Song, Sam H. Noh, Jongryool Kim"
date: 2026-07-21
tags: ["serving", "kv-cache", "memory-hierarchy"]
topic: 'serving'
summary: "HyMCache stages KV-cache blocks in a CXL-hybrid memory tier (a little device DRAM in front of large SSD-backed capacity) for multi-turn LLM serving, using request-level prefix prefetching to exploit the read-dominant, append-only nature of multi-turn access, reaching 3.0x the throughput of local LMCache at a matched DRAM budget while using 16x less DRAM than a 1TB distributed-DRAM Mooncake deployment."
summary_ko: "HyMCache는 멀티턴 LLM 서빙을 위해 CXL-하이브리드 메모리 계층(소량의 디바이스 DRAM + 대용량 SSD)에 KV 캐시 블록을 스테이징하고, 멀티턴 접근의 읽기-지배적·추가전용(append-only) 특성을 활용하는 요청 단위 프리픽스 프리페칭으로, 동일 DRAM 예산에서 로컬 LMCache 대비 3.0배 처리량을 내면서 1TB 분산-DRAM Mooncake 배포 대비 DRAM을 16배 적게 쓴다."
links: ["sangam", "kivi"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.18141' }
figures:
  - src: /figures/hymcache/fig1.png
    caption: "HyMCache overview and walkthrough: request-level prefix prefetching and opportunistic write buffering stage latency-critical KV reads in device DRAM ahead of time, while bulk capacity stays SSD-backed behind the CXL interface."
    caption_ko: "HyMCache 개요와 동작 예시: 요청 단위 프리픽스 프리페칭과 기회적 쓰기 버퍼링이 지연에 민감한 KV 읽기를 디바이스 DRAM에 미리 스테이징하고, 대용량 데이터는 CXL 인터페이스 뒤편의 SSD에 남는다."
    credit: "Figure 7 from arXiv:2607.18141 — authors' figure"
  - src: /figures/hymcache/fig2.png
    caption: "End-to-end single-node serving performance: HyMCache against recomputation, GPU prefix caching, and local-DRAM LMCache baselines."
    caption_ko: "종단 간 단일 노드 서빙 성능: HyMCache와 재계산, GPU 프리픽스 캐싱, 로컬-DRAM LMCache 베이스라인 비교."
    credit: "Figure 11 from arXiv:2607.18141 — authors' figure"
analysis:
  ko:
    background: '장문맥·멀티턴·에이전틱 LLM 워크로드는 이전에 처리한 컨텍스트를 계속 재사용하기 때문에 KV 캐시 재사용이 중복 연산을 줄이는 핵심 수단이 된다. 그런데 이 재사용은 병목을 연산에서 "재사용 가능한 KV 상태를 저장하고 서빙하는 메모리 계층"으로 옮겨 놓는다.'
    problem: 'GPU HBM과 호스트 DRAM은 클러스터 규모에서 TB급 공유 컨텍스트 용량으로 확장하기에는 비용이 너무 크다. 더 값싸고 용량이 큰 원격 매체로 만든 계층이 필요하지만, 그런 매체를 멀티턴 KV 캐시라는 특정 워크로드에 맞게 어떻게 관리할지가 미해결 문제다.'
    prior_limits: '분산-DRAM 기반 KV 캐시 시스템(예: 1TB DRAM을 쓰는 Mooncake)은 성능은 좋지만 순수 DRAM 용량 확장 비용을 그대로 치른다. CXL 기반 계층형 메모리(CMM-H) 장치는 일반적으로 내부 DRAM을 투명한 LRU 캐시로 취급하는데, 이는 멀티턴 KV 캐시 접근이 갖는 예측 가능한 순서를 활용하지 못한다.'
    goal: '내부 DRAM을 단순한 투명 캐시가 아니라 명시적으로 관리되는 스테이징 공간으로 재설계해, SSD 기반 대용량 원격 KV 재사용에서도 DRAM 수준의 캐시 효율을 SSD 수준의 비용으로 얻는 것이 목표다.'
    method: '**HyMCache**는 멀티턴 LLM 서빙을 위해 CXL-하이브리드 메모리(CXL-HM: 소량의 디바이스 DRAM + 대용량 SSD 기반 원격 계층)를 사용하는 KV 캐시 프레임워크다. 멀티턴 프리픽스 캐시 조회가 드러내는 예측 가능한 KV 객체 접근 순서를 활용해, 요청 단위 프리픽스 프리페칭과 기회적 쓰기 버퍼링으로 지연에 민감한 읽기를 미리 디바이스 DRAM에 올려놓는다. 저녁 예약 시간을 미리 알고 손님이 도착하기 전에 발레파킹 요원이 차 열쇠를 미리 꺼내 놓는 것과 비슷하다 — 접근 순서가 예측 가능하기 때문에 필요해지기 전에 준비해 둘 수 있다.'
    key_idea: '핵심은 멀티턴 KV 캐시 접근이 갖는 세 가지 성질 — 읽기-지배적(read-dominant), 예측 가능(predictable), 추가전용(append-only) — 을 명시적으로 활용해 일반적인 투명 LRU 캐시보다 훨씬 똑똑하게 디바이스 DRAM을 관리하는 것이다. 이 성질들 덕분에 "다음에 무엇이 필요할지"를 사후적으로 추측하는 대신 사전에 알 수 있다.'
    validation: '실제 CXL-HM 프로토타입 위에서 단일-집계(single-aggregator) 서빙과 PD-분리(prefill-decode-disaggregated) 서빙 두 구성 모두를 평가한다. 동일 DRAM 예산 아래 로컬 LMCache, 그리고 1TB 분산-DRAM을 쓰는 Mooncake와 비교한다.'
    results: '동일 DRAM 예산에서 HyMCache는 단일 노드 서빙에서 로컬 LMCache 대비 **3.0배**, PD-분리 서빙에서 **1.45배** 성능을 낸다. 1TB 분산-DRAM Mooncake와 비교하면 성능은 약 **30% 낮지만** DRAM은 **16배 적게** 쓴다 — DRAM 1GB당 효율로 보면 크게 유리한 트레이드오프다.'
    comparison: 'Mooncake와의 비교가 이 논문의 포지셔닝을 가장 잘 보여준다 — HyMCache는 최고 성능을 노리지 않고, 압도적으로 적은 DRAM으로 TB급 KV 재사용 용량을 SSD 수준 비용에 얻는 것을 목표로 한다. 로컬 LMCache와 비교해서는 순수하게 이긴다(같은 DRAM 예산에서 3.0배/1.45배).'
    significance: '서빙 스택 설계에서 "KV 캐시는 GPU HBM이나 호스트 DRAM에 있어야 빠르다"는 암묵적 전제에, CXL 기반 원격 SSD 계층도 접근 패턴을 제대로 활용하면 실용적 대안이 될 수 있음을 보여준다. TB급 공유 컨텍스트가 필요한 장문맥·멀티턴·에이전틱 서빙 클러스터에 특히 관련이 크다.'
    limitations: '논문 자체의 결론부에는 별도 Limitations 절이 없다. 평가가 특정 CXL-HM 하드웨어 프로토타입과 Qwen2.5-32B 등 테스트된 모델 규모에 한정된다는 점, 그리고 "읽기-지배적·예측 가능·추가전용"이라는 핵심 가정이 깨지는 워크로드(예: 단발성 질의나 매우 불규칙한 멀티턴 패턴)에서의 동작은 다뤄지지 않는다는 점은 리뷰어가 덧붙인다(리뷰어 판단).'
    future_work: '논문 내 명시된 별도 future-work 절은 없다. Conclusion은 실제 CXL-HM 프로토타입에서 단일-집계와 PD-분리 서빙 모두에 대해 HyMCache가 미래 LLM 서빙 시스템을 위한 확장 가능하고 비용 효율적인 원격 KV 캐시 용량을 가능케 한다는 결과 요약으로 끝난다.'
    resources: '논문 본문에 인용된 GitHub 링크(NIXL, aiperf)는 모두 저자들이 사용한 제3자 도구에 대한 인용일 뿐, 이 논문 자체의 코드 저장소가 아니다 — 공개 링크 확인 안 됨.'
  en:
    background: 'Long-context, multi-turn, and agentic LLM workloads keep reusing previously processed context, which makes KV-cache reuse essential for cutting redundant computation. That reuse, however, shifts the bottleneck from compute to the memory tier that stores and serves reusable KV state.'
    problem: 'GPU HBM and host DRAM are too costly to scale to TB-scale shared-context capacity at cluster scale. A tier built from cheaper, higher-capacity remote media is needed, but how to manage such media specifically for the multi-turn KV-cache workload is an open question.'
    prior_limits: 'Distributed-DRAM KV-cache systems (e.g. Mooncake with 1TB of DRAM) perform well but pay the full cost of scaling pure DRAM capacity. CXL-based tiered-memory devices (CMM-H) typically treat their internal DRAM as a transparent LRU cache, which fails to exploit the predictable access order that multi-turn KV-cache access actually has.'
    goal: 'Redesign the internal DRAM as an explicitly managed staging space rather than a transparent cache, so that SSD-backed, TB-scale remote KV reuse can achieve DRAM-level cache efficiency at SSD-level cost.'
    method: '**HyMCache** is a KV-cache framework for multi-turn LLM serving that uses CXL-hybrid memory (CXL-HM: a small amount of in-device DRAM plus large SSD-backed remote capacity). It exploits the predictable KV-object access order revealed by multi-turn prefix-cache lookups, using request-level prefix prefetching and opportunistic write buffering to stage latency-critical reads in device DRAM ahead of time. It is like a valet who already knows a dinner reservation time and pulls the car keys out in advance — because the access order is predictable, the system can prepare before the need arrives.'
    key_idea: 'The core idea is to explicitly exploit three properties of multi-turn KV-cache access — read-dominant, predictable, and append-only — to manage device DRAM far more intelligently than a generic transparent LRU cache. Those properties mean the system can know what will be needed next in advance, rather than guessing after the fact.'
    validation: 'Evaluated on a real CXL-HM prototype under both single-aggregator and PD-disaggregated (prefill-decode-disaggregated) serving configurations, compared against local LMCache under the same DRAM budget and against Mooncake using 1TB of distributed DRAM.'
    results: 'Under the same DRAM budget, HyMCache outperforms local LMCache by **3.0x** in single-node serving and **1.45x** in PD-disaggregated serving. Compared with a 1TB distributed-DRAM Mooncake deployment, it incurs about **30% lower** performance but uses **16x less** DRAM — a strongly favorable trade-off in per-GB-of-DRAM efficiency.'
    comparison: 'The comparison against Mooncake best captures this paper''s positioning — HyMCache isn''t chasing peak performance, it''s aiming for TB-scale KV-reuse capacity at SSD-level cost using dramatically less DRAM. Against local LMCache, it wins outright (3.0x/1.45x at the same DRAM budget).'
    significance: 'Challenges the implicit assumption in serving-stack design that KV caches "must" live in GPU HBM or host DRAM to be fast — a CXL-based remote SSD tier can be a practical alternative if the access pattern is genuinely exploited. Particularly relevant to long-context, multi-turn, and agentic serving clusters that need TB-scale shared context.'
    limitations: 'The paper has no dedicated Limitations section in its conclusion. That the evaluation is confined to a specific CXL-HM hardware prototype and tested model scales (e.g. Qwen2.5-32B), and that behavior under workloads violating the core "read-dominant, predictable, append-only" assumption (single-turn queries, highly irregular multi-turn patterns) is not addressed, are reviewer additions (reviewer judgment).'
    future_work: 'Not stated in the paper. The Conclusion ends by summarizing that, evaluated on a real CXL-HM prototype under both single-aggregator and PD-disaggregated serving, HyMCache enables scalable and cost-efficient remote KV-cache capacity for future LLM serving systems.'
    resources: 'The GitHub links cited in the paper (NIXL, aiperf) are references to third-party tools the authors used, not this paper''s own code repository — no public release verified.'
thread:
  ko: |-
    KV 캐시를 메모리 계층에 걸쳐 어떻게 배치할 것인가는 서빙 시스템 문헌의 오래된 질문이다 — GPU 프리픽스 캐싱은 가장 빠르지만 가장 비싸고, Mooncake 같은 분산-DRAM 시스템은 그 다음으로 빠르면서 DRAM 확장 비용을 그대로 치른다. CXL 기반 계층형 메모리는 더 값싼 대안으로 등장했지만, 대부분의 CMM-H 장치는 내부 DRAM을 범용 투명 LRU 캐시로 다뤄 특정 워크로드의 구조를 활용하지 못했다.

    HyMCache의 개념적 전환은 "범용 캐시"라는 틀을 버리고 멀티턴 KV 캐시라는 특정 워크로드의 접근 패턴 — 읽기-지배적, 예측 가능, 추가전용 — 을 정면으로 활용하는 것이다. 일반적인 LRU가 과거 접근 이력으로 미래를 추측한다면, HyMCache는 멀티턴 프리픽스 캐시 조회 구조 자체가 미래의 접근 순서를 사실상 미리 알려준다는 점을 이용해 사전에 스테이징한다. 이는 "저비용 원격 매체 vs 캐시 효율"이라는 트레이드오프를 완전히 없애지는 못하지만, 워크로드 구조를 활용해 그 트레이드오프의 곡선 자체를 유리하게 옮긴다.

    이 논문이 여는 다음 질문은 이 워크로드-특화 스테이징 원칙이 얼마나 일반화되는가이다. 멀티턴 대화나 에이전틱 루프처럼 접근이 예측 가능한 워크로드를 넘어, 더 불규칙한 서빙 패턴이나 MoE처럼 KV 블록 크기와 접근 패턴이 크게 다른 아키텍처에도 같은 원칙이 통할지는 이 논문이 다루지 않는다.
  en: |-
    How to place the KV cache across the memory hierarchy is a long-standing question in the serving-systems literature — GPU prefix caching is fastest but most expensive, distributed-DRAM systems like Mooncake are the next-fastest tier while still paying the cost of DRAM scaling. CXL-based tiered memory emerged as a cheaper alternative, but most CMM-H devices treat their internal DRAM as a generic transparent LRU cache, failing to exploit the structure of any particular workload.

    HyMCache's conceptual shift is to drop the "generic cache" framing and directly exploit the access pattern specific to multi-turn KV-cache workloads — read-dominant, predictable, append-only. Where a generic LRU guesses the future from past access history, HyMCache stages ahead of time by exploiting the fact that multi-turn prefix-cache lookup structure effectively reveals the future access order. That doesn't eliminate the trade-off between cheap remote media and cache efficiency, but it shifts the trade-off curve itself in a favorable direction by leveraging workload structure.

    The question this paper leaves open is how far this workload-specific staging principle generalizes. Beyond workloads with predictable access — multi-turn dialogue, agentic loops — the paper doesn't explore whether the same principle holds for more irregular serving patterns, or for architectures like MoE where KV block sizes and access patterns can differ substantially from the dense models tested here.
sparks:
  - ko: '논문이 명시적으로 기대는 세 가지 접근 패턴 가정(읽기-지배적, 예측 가능, 추가전용)이 깨지는 워크로드 — 예를 들어 단발성 질의가 지배적이거나 접근 순서가 매우 불규칙한 서빙 — 에서는 HyMCache의 이득이 얼마나 남을까?'
    en: "How much of HyMCache's benefit survives in workloads where its three explicit access-pattern assumptions (read-dominant, predictable, append-only) break down — dominated by single-turn queries, say, or serving with highly irregular access order?"
  - ko: '평가는 Qwen2.5-32B 같은 밀집(dense) 모델에 집중한다 — KV 블록 크기와 접근 패턴이 크게 다른 MoE 아키텍처나, 더 큰 모델 규모에서도 같은 스테이징 전략이 통할지는 논문이 다루지 않는다.'
    en: "The evaluation focuses on dense models like Qwen2.5-32B — whether the same staging strategy holds for MoE architectures (where KV block sizes and access patterns can differ substantially) or at larger model scales is not explored in the paper."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
