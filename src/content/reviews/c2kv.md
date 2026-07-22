---
title: "C²KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference"
arxivId: "2607.17715"
authors: "Chuheng Du, Junyi Chen, Hanlin Tang, Kan Liu, Tao Lan, Lin Qu, Chaoyue Niu, Shengzhong Liu, Guihai Chen, Fan Wu"
date: 2026-07-22
tags: ["kv-cache", "compression", "serving"]
topic: 'kv-cache'
summary: "C2KV is a KV-cache reuse framework for long-context LLM inference that combines compression with non-prefix reuse via a lightweight sidecar Extractor and a structured attention flow, trained with a compression-concatenation co-training strategy, reaching up to 17x inference acceleration without modifying the base model."
summary_ko: "C2KV는 경량 사이드카 Extractor와 구조화된 attention flow로 압축과 비-프리픽스(non-prefix) KV 캐시 재사용을 결합한 장문맥 LLM 추론 프레임워크로, compression-concatenation 공동학습 전략을 통해 베이스 모델 수정 없이 최대 17배 추론 가속을 달성한다."
links: ["kivi", "kvquant", "hymcache"]
resources:
  - label: 'arXiv'
    url: 'https://arxiv.org/abs/2607.17715'
  - label: 'GitHub'
    url: 'https://github.com/s7a9/C2KV'
figures:
  - src: /figures/c2kv/fig1.png
    caption: "Architecture overview of C2KV: a lightweight sidecar Extractor produces position-agnostic, composable KV representations via a structured attention flow, without modifying the base model."
    caption_ko: "C2KV 아키텍처 개요: 경량 사이드카 Extractor가 구조화된 attention flow를 통해 베이스 모델 수정 없이 위치-독립적이고 조합 가능한 KV 표현을 생성한다."
    credit: "Figure 5 from arXiv:2607.17715 — authors' figure"
  - src: /figures/c2kv/fig2.png
    caption: "Trade-off between Time-to-First-Token (TTFT) and task accuracy under non-prefix document reuse, comparing C2KV against alternative reuse strategies."
    caption_ko: "비-프리픽스 문서 재사용 환경에서 TTFT(첫 토큰까지 시간)와 과제 정확도 사이의 트레이드오프를 C2KV와 다른 재사용 전략들이 비교한 결과."
    credit: "Figure 7 from arXiv:2607.17715 — authors' figure"
analysis:
  ko:
    background: '긴 문맥을 반복적으로 다시 쓰는 인컨텍스트 러닝(ICL)이나 RAG 같은 워크로드에서는, 매번 처음부터 프리필(prefill)하는 대신 이미 계산해 둔 KV 캐시를 재사용해 첫 토큰까지의 시간(TTFT)을 줄이려는 시도가 늘고 있다. 문제는 재사용하려는 문서 조각들이 매 요청마다 다른 순서·조합으로 등장하는 비-프리픽스(non-prefix) 상황이 흔하다는 점이다.'
    problem: 'KV 캐시를 압축해서 저장·전송 비용을 줄이는 것과, 프리픽스가 아닌 임의 조합으로 캐시를 이어붙여 재사용하는 것을 그냥 순진하게 합치면 정확도가 심각하게 무너진다. 즉 압축과 비-프리픽스 재사용은 각각은 잘 작동해도 함께 쓰면 서로를 방해한다.'
    prior_limits: '기존 KV 재사용 기법들은 대체로 프리픽스 공유(같은 접두사를 가진 요청들끼리만 캐시 공유)를 전제하거나, 압축을 걸더라도 재사용 시나리오에서의 위치 불일치·attention 패턴 붕괴를 별도로 다루지 않았다. KV 편차(deviation)는 recompute 비율이 낮아질수록 커진다는 것이 이 논문이 지적하는 구체적 실패 양상이다.'
    goal: '베이스 모델을 전혀 수정하지 않으면서, 압축된 KV 표현을 어떤 순서로 이어붙이더라도(비-프리픽스) 정확도 손실 없이 재사용 가능하게 만드는 것이 목표다.'
    method: '경량 사이드카 Extractor가 학습 가능한 압축 토큰(compression token)들을 사용해 원본 KV를 압축된 표현으로 뽑아내고, 구조화된 attention flow가 이 압축 표현들을 조합할 때 위치 정보를 일관되게 유지시킨다. 학습은 compression-concatenation 공동학습 전략을 쓰는데, 압축과 재사용(연결) 두 과정을 따로 학습하지 않고 함께 최적화해 둘 사이의 불일치를 학습 단계에서부터 줄인다. 마치 여러 사람이 각자 요약한 회의록 조각을 나중에 순서 상관없이 이어 붙여도 말이 되도록, 요약할 때부터 이어붙이기를 염두에 두고 요약하는 것과 비슷하다.'
    key_idea: '핵심은 "압축"과 "재사용 시의 조합 가능성(composability)"을 별개의 단계로 처리하지 않고 하나의 학습 목표로 묶은 것이다. Extractor가 만드는 표현은 처음부터 위치-독립적으로 설계되어, 어떤 순서로 이어붙여도 구조화된 attention flow가 이를 일관되게 소화할 수 있다.'
    validation: '장문맥 인컨텍스트 러닝 워크로드에서 TTFT와 디코드 지연시간을 측정하고, RULER 벤치마크로 여러 문맥 길이·답변 깊이(answer depth) 조합에서의 정확도를 평가한다. blending과 loading 두 재사용 방식의 지연시간 비교, 문맥 길이에 따른 디코드 지연시간 비교 등 시스템 차원의 검증도 포함한다.'
    results: '장문맥 시나리오에서 최대 **17배** 추론 가속을 보고하며, 생성 품질은 유지하면서 KV 캐시 저장·전송 비용을 줄인다고 주장한다. RULER 평가에서는 4배 압축률로 테스트된다.'
    comparison: '이 리뷰가 참조한 초록 수준에서는 구체적으로 어떤 베이스라인 대비 17배인지(단순 재계산 대비인지, 다른 압축·재사용 기법 대비인지)는 명확히 확인되지 않는다 — 논문 내 명시 없음(이 리뷰 범위 기준). TTFT-정확도 트레이드오프 곡선에서 C2KV가 다른 재사용 전략들보다 우수한 프론티어를 그린다고 보고된다.'
    significance: '효율적 LLM 서빙 관점에서, 이 논문은 "압축이냐 재사용이냐"가 아니라 "압축과 재사용을 함께 학습해야 한다"는 점을 명시적으로 겨냥한다. RAG나 멀티-문서 ICL처럼 문맥 조각을 매번 다른 조합으로 이어붙이는 워크로드가 늘어나는 상황에서, 프리픽스 캐싱만으로는 부족한 지점을 채우는 실용적 기여다.'
    limitations: '압축률이 매우 높아질 때(4배를 넘어서는 영역) 정확도가 어떻게 무너지는지, 그리고 베이스 모델과 무관한 사이드카 구조가 서로 다른 아키텍처(예: MLA, 하이브리드 어텐션)에도 그대로 이식되는지는 이 리뷰가 참조한 초록 수준에서 명시되지 않는다(리뷰어 판단).'
    future_work: '논문 내 명시된 future-work 항목은 이 리뷰가 참조한 초록 범위에서는 확인되지 않는다.'
    resources: '공식 GitHub 저장소가 확인된다(s7a9/C2KV, curl로 200 응답 확인) — 코드가 공개되어 있다. ACM SIGKDD 2026 채택 논문이다.'
  en:
    background: 'Workloads that repeatedly reuse long context, such as in-context learning (ICL) or RAG, increasingly try to avoid re-prefilling from scratch every time by reusing an already-computed KV cache to cut time-to-first-token (TTFT). The catch is that the document chunks being reused often appear in a different order or combination on every request, a non-prefix setting.'
    problem: "Naively combining KV-cache compression with non-prefix reuse, gluing together compressed cache fragments in arbitrary order, causes severe accuracy degradation. Compression alone works, and prefix-based reuse alone works, but stacking them naively breaks accuracy."
    prior_limits: 'Prior KV reuse techniques largely assume prefix sharing (only requests sharing the same prefix can share cache), and even when compression is applied, they don''t separately address the positional mismatch and attention-pattern breakdown that shows up specifically in reuse scenarios. The paper''s concrete failure mode is that KV deviation grows as the recompute ratio drops.'
    goal: 'Without modifying the base model at all, make compressed KV representations reusable in any concatenation order (non-prefix) without an accuracy penalty.'
    method: "A lightweight sidecar Extractor uses learnable compression tokens to extract compressed representations from the original KV, while a structured attention flow keeps positional information consistent when these compressed representations are concatenated in arbitrary order. Training uses a compression-concatenation co-training strategy: compression and reuse (concatenation) are not trained as separate stages but optimized jointly, closing the gap between them at training time rather than papering over it at inference time. It is a bit like several people each summarizing a piece of a meeting, with the summaries written from the start to make sense when stitched together in any order, rather than summarized independently and reconciled afterward."
    key_idea: "The core move is treating compression and reuse-time composability not as separate stages but as one joint training objective. The Extractor's representations are designed to be position-agnostic from the start, so the structured attention flow can consume them consistently regardless of concatenation order."
    validation: 'TTFT and decode latency are measured on long-context in-context-learning workloads, and accuracy is evaluated on the RULER benchmark across multiple context lengths and answer depths. System-level validation also includes a latency comparison between blending and loading reuse strategies and a decode-latency comparison across context lengths.'
    results: 'Reports up to **17x** inference acceleration in long-context scenarios, claiming reduced KV-cache storage and transfer cost while maintaining generation quality. RULER evaluation is run at a 4x compression ratio.'
    comparison: 'At the abstract-level scope this review draws from, exactly which baseline the 17x figure is measured against (naive recomputation vs. other compression-and-reuse methods) is not clearly identifiable, not stated in the paper (from this review''s vantage point). The TTFT-vs-accuracy trade-off curve is reported to place C2KV on a better frontier than alternative reuse strategies.'
    significance: 'From an efficient-LLM-serving standpoint, this paper explicitly targets not "compression or reuse" but "compression and reuse must be trained together." As workloads like RAG or multi-document ICL that stitch context fragments in ever-different combinations become more common, this is a practical contribution to a gap prefix caching alone cannot fill.'
    limitations: 'How accuracy degrades at very high compression ratios (beyond the 4x tested), and whether the base-model-agnostic sidecar design transfers as-is to different architectures (e.g. MLA, hybrid attention), are not stated at the abstract-level scope this review draws from (reviewer judgment).'
    future_work: 'No explicit future-work items are identifiable within the abstract-level scope this review draws from.'
    resources: 'An official GitHub repository is verified (s7a9/C2KV, confirmed via curl, HTTP 200), the code is publicly released. Accepted at ACM SIGKDD 2026.'
thread:
  ko: |-
    KV 캐시 재사용 연구는 대체로 프리픽스 공유라는 좁은 전제에서 출발했다 — vLLM류 시스템의 RadixAttention이나 prefix caching이 대표적으로, 같은 접두사를 가진 요청들끼리만 캐시를 나눠 쓴다. 압축 쪽 연구(KIVI, KVQuant 등)는 이와 별개로 발전해왔고, 두 갈래가 만나는 지점, 즉 "압축된 캐시를 프리픽스가 아닌 임의 조합으로도 재사용한다"는 조합은 상대적으로 덜 탐구된 영역이었다.

    C2KV의 전환은 이 두 갈래를 그냥 이어붙이지 않고 처음부터 하나의 학습 문제로 재정의한 데 있다. 압축을 먼저 하고 재사용을 나중에 붙이는 파이프라인 방식 대신, 압축된 표현 자체가 "어떤 순서로 이어붙여도 말이 되도록" 설계되고 학습되어야 한다는 것이다. 이는 압축률과 재사용 유연성을 별개의 트레이드오프로 보지 않고, 학습 목표 설계로 둘을 동시에 다루는 접근이다.

    이 논문이 여는 질문은 이 조합-가능성(composability) 원리가 압축률의 한계와 어떻게 맞물리는가이다 — 4배 압축에서는 잘 작동한다고 보고되지만, 훨씬 공격적인 압축률에서도 위치-독립적 표현이라는 설계가 버텨줄지, 그리고 이 사이드카 방식이 MLA나 하이브리드 어텐션처럼 KV 구조 자체가 다른 아키텍처에도 그대로 이식되는지는 다뤄지지 않는다.
  en: |-
    KV-cache reuse research has largely started from the narrow premise of prefix sharing, systems like vLLM's RadixAttention or prefix caching split cache only among requests sharing the same prefix. Compression research (KIVI, KVQuant, and others) has developed on a separate track, and the intersection of the two, compressed caches reused in arbitrary, non-prefix combinations, has been a comparatively underexplored corner.

    C2KV's shift is refusing to simply bolt the two tracks together and instead redefining them as one training problem from the start. Rather than a pipeline where compression happens first and reuse is stitched on afterward, the compressed representation itself must be designed and trained to "make sense however it's concatenated." This treats compression ratio and reuse flexibility not as a separate trade-off but as something a single training objective can address jointly.

    The question this paper leaves open is how this composability principle interacts with the limits of compression ratio, it is reported to work well at 4x compression, but whether the position-agnostic representation design holds up at much more aggressive ratios, and whether this sidecar approach transfers as-is to architectures with a fundamentally different KV structure (MLA, hybrid attention), are both left unaddressed.
sparks:
  - ko: '4배보다 훨씬 공격적인 압축률에서 위치-독립적 표현 설계가 정확도를 얼마나 지켜낼 수 있는지는 논문이 직접 다루지 않는 열린 질문이다.'
    en: "How well the position-agnostic representation design holds accuracy at compression ratios far more aggressive than 4x is an open question the paper doesn't directly address."
  - ko: 'MLA나 하이브리드 어텐션처럼 KV 구조 자체가 표준 GQA/MHA와 다른 아키텍처에 이 사이드카 Extractor 접근을 그대로 적용하면 어떤 조정이 필요할지는 논문의 범위 밖이다.'
    en: "What adjustments this sidecar-Extractor approach would need to apply to architectures with a fundamentally different KV structure than standard GQA/MHA, such as MLA or hybrid attention, is outside this paper's scope."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
