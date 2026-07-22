---
title: "LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Models"
arxivId: "2607.16339"
authors: "Xingru Chen, Zelang Liang, Yongjia Ma, Jiqing Zhan, Shuling Yang, Lian Wen, Kun Zhan"
date: 2026-07-22
tags: ["dllm", "caching", "quantization", "acceleration"]
topic: 'diffusion-llm'
summary: "LaCache is a training-free acceleration framework for diffusion LLMs that eliminates redundant denoising-step computation via exact intermediate-result caching and adds per-group FP8 quantization to FFN layers, reaching 1.3x speedup alone and up to 40.2x when stacked with existing acceleration methods."
summary_ko: "LaCache는 확산 언어모델의 디노이징 스텝 중복 계산을 무손실 중간결과 캐싱으로 제거하고 FFN 레이어에 그룹별 FP8 양자화를 더한 학습-불필요 가속 프레임워크로, 단독으로 1.3배, 기존 가속 기법과 결합 시 최대 40.2배 속도 향상을 낸다."
links: ["llada", "fast-dllm", "flowblock"]
resources:
  - label: 'arXiv'
    url: 'https://arxiv.org/abs/2607.16339'
figures:
  - src: /figures/lacache/fig1.png
    caption: "Left: block-diffusion dLLMs redo computation on tokens outside the active block at every denoising step even though those tokens are unchanged, causing repeated redundant compute. Right: LaCache's Lossless State Memoization caches those invariant intermediate results instead."
    caption_ko: "왼쪽: 블록-확산 dLLM은 활성 블록 밖의 변하지 않는 토큰에 대해서도 디노이징 스텝마다 계산을 반복해 중복 연산이 발생한다. 오른쪽: LaCache의 Lossless State Memoization은 이 불변 중간결과를 캐싱해 재계산을 없앤다."
    credit: "Figure 1 from arXiv:2607.16339 — authors' figure"
  - src: /figures/lacache/fig2.png
    caption: "Average inference latency of LaCache vs. prior acceleration methods on LLaDA-Instruct across multiple benchmarks — LaCache further reduces latency on top of both parallel decoding and DPad."
    caption_ko: "LLaDA-Instruct에서 여러 벤치마크에 걸친 LaCache와 기존 가속 기법의 평균 추론 지연시간 비교 — LaCache는 parallel decoding, DPad 위에서도 지연시간을 추가로 줄인다."
    credit: "Figure 5 from arXiv:2607.16339 — authors' figure"
analysis:
  ko:
    background: '블록-확산 대형언어모델(dLLM)은 시퀀스를 블록으로 나누고 각 블록 안에서 여러 디노이징 스텝에 걸쳐 병렬로 토큰을 채워 넣는다. 문제는 한 블록을 생성하는 동안 그 블록 밖의 프리픽스와 마스크된 서픽스는 값이 전혀 바뀌지 않는데도, 표준 구현은 매 스텝 전체 시퀀스를 다시 계산한다는 점이다.'
    problem: '이 반복 재계산은 근사나 정확도 손실 없이도 제거 가능한 순수한 연산-수준 중복이다. 질문은 정확도를 전혀 건드리지 않고 이 중복을 얼마나 없앨 수 있는가이며, 여기에 더해 메모리 대역폭 병목까지 함께 줄일 수 있는가이다.'
    prior_limits: '기존 dLLM 가속 기법들(병렬 디코딩, DPad 등)은 디노이징 스텝 수 자체를 줄이거나 스텝 내 토큰 처리를 병렬화하는 데 집중해왔고, 스텝들 사이에서 반복되는 임베딩·RoPE·attention 중간결과의 무손실 재사용은 별도로 다루지 않았다 — 서로 직교하는 최적화 축이다.'
    goal: '추가 학습이나 근사 없이, 디노이징 스텝 사이에서 불변인 중간결과를 정확히(losslessly) 캐싱해 재계산을 스킵하고, 여기에 메모리 대역폭 병목을 더 줄이는 정밀도 조정을 얹는 것이 목표다.'
    method: 'Lossless State Memoization(LSM)은 세 가지를 캐싱한다: 임베딩 출력을 위한 EmbedCache, 토큰별 pre-attention 상태를 위한 RoPECache, FlashAttention 내부의 온라인 softmax 통계를 위한 FACache. 세 캐시 모두 값이 바뀌지 않는 토큰에 대한 재계산을 건너뛰게 해주며, 출력은 원래 계산과 정확히 동일하다. 여기에 더해, 디노이징 단계마다 활성화 분포가 달라지는 FFN 레이어에는 별도로 그룹별 FP8 양자화를 적용한다. 이미 인쇄되어 나온 신문 페이지에서 바뀐 문단만 다시 조판하고 나머지는 그대로 재사용하는 것과 비슷하다.'
    key_idea: '핵심은 무손실 중복 제거와 손실 있는 정밀도 조정을 서로 다른 텐서·서로 다른 축에 분리해서 적용한 것이다 — LSM은 임베딩/RoPE/attention 중간결과라는 캐시 가능한 불변 항목에, FP8은 스텝마다 분포가 흔들리는 FFN 활성화에만 건다. 두 최적화가 서로의 정확도 손실을 키우지 않도록 아예 다른 대상을 겨냥한 설계다.'
    validation: 'LLaDA-Instruct, LLaDA-base, LLaDA-1.5, Fast-dLLM-V2 등 여러 백본과 다수의 수학·코드·일반 벤치마크에서 정확도와 지연시간을 측정한다. 캐싱 전략 단독의 ablation, 양자화 그래뉼래러티별 정확도·효율 비교, 캐시 갱신 주기에 따른 정확도-속도 트레이드오프 곡선, 문맥 길이별 ablation까지 포함한 폭넓은 검증이다.'
    results: 'LaCache 단독으로 원본 dLLM 대비 약 **1.3배** end-to-end 속도 향상을 내고, 기존 가속 기법들과 결합하면 최대 **40.2배**까지 올라가며, 이 과정에서 과제 정확도는 유사한 수준으로 유지된다고 보고한다. 긴 문맥 추론(HumanEval)에서도 토큰 생성 속도가 추가로 개선된다.'
    comparison: '병렬 디코딩이나 DPad 같은 기존 가속 기법 위에 LaCache를 얹으면 각 기법 단독보다 지연시간이 더 줄어든다고 보고된다 — 즉 스텝-수 감소 계열 기법과 경쟁하는 대신 그 위에 곱으로 쌓이는 위치를 차지한다. FP8 FFN 양자화는 K/V 캐시가 아니라 FFN에만 적용되므로, K/V 캐시 자체를 압축하는 기법들과는 대상 텐서가 겹치지 않는다.'
    significance: '효율적 AI 관점에서 이 논문은 dLLM 가속에는 아직 공짜 점심(무손실 중복 제거)이 남아 있다는 것을 보여준다 — 근사나 품질 저하 없이 얻을 수 있는 1.3배는 이후에 얹는 모든 손실 압축 기법의 순수한 보너스가 된다. 특히 스텝-수 축소 기법들과 직교적으로 결합 가능하다는 점이 실서빙 스택 설계에 실용적이다.'
    limitations: '어떤 dLLM 백본(SDAR, TraDo 등 블록-확산 계열 전반)까지 일반화되는지, 그리고 블록 안 프리픽스/서픽스 불변이라는 LSM의 핵심 가정이 토큰이 확정된 뒤 다시 수정(remask/revocation)되는 경우에도 안전한지는 논문 초록 수준에서 명시되지 않는다(리뷰어 판단).'
    future_work: '논문 내 명시된 future-work 항목은 이 리뷰가 참조한 초록 범위에서는 확인되지 않는다 — 전문(全文)의 결론부 확인이 필요하다.'
    resources: '공개 코드 저장소는 확인되지 않았다 — 공개 링크 확인 안 됨.'
  en:
    background: 'Block-diffusion large language models (dLLMs) split a sequence into blocks and fill in tokens in parallel across multiple denoising steps within each block. The catch is that while one block is being generated, the prefix before it and the still-masked suffix after it never change value, yet a standard implementation recomputes the entire sequence at every single step anyway.'
    problem: 'This repeated recomputation is pure operator-level redundancy, removable without any approximation or accuracy cost. The question is how much of it can be eliminated while touching accuracy not at all, and whether the memory-bandwidth bottleneck can be cut at the same time.'
    prior_limits: 'Existing dLLM acceleration methods (parallel decoding, DPad, etc.) focus on reducing the number of denoising steps or parallelizing token processing within a step; none of them separately address the lossless reuse of embedding, RoPE, and attention intermediates that repeat identically across steps, an orthogonal optimization axis.'
    goal: 'Without any additional training or approximation, exactly (losslessly) cache the intermediate results that stay invariant across denoising steps to skip recomputation, then layer a precision adjustment on top to further cut the memory-bandwidth bottleneck.'
    method: "Lossless State Memoization (LSM) caches three things: EmbedCache for embedding outputs, RoPECache for token-wise pre-attention states, and FACache for FlashAttention's internal online-softmax running statistics. All three let the model skip recomputation for tokens whose values haven't changed, with output identical to the uncached computation. On top of this, LaCache applies per-group FP8 quantization specifically to FFN layers, tuned to how FFN activation distributions shift step-by-step across the diffusion process. It is a bit like re-typesetting only the paragraphs that changed on an already-printed newspaper page, and reusing the rest as-is."
    key_idea: "The core move is separating lossless redundancy elimination from lossy precision adjustment and applying each to a different tensor on a different axis. LSM targets the cacheable, invariant embedding/RoPE/attention intermediates, while FP8 targets only the FFN activations whose distribution actually shifts step-to-step. Keeping the two optimizations aimed at disjoint targets means neither compounds the other's accuracy cost."
    validation: 'Accuracy and latency are measured across multiple backbones (LLaDA-Instruct, LLaDA-base, LLaDA-1.5, Fast-dLLM-V2) and a broad set of math, code, and general benchmarks. The validation includes ablations of the caching strategy alone, accuracy/efficiency comparisons across quantization granularities, an accuracy-vs-speed curve as a function of cache update interval, and context-length ablations.'
    results: 'LaCache alone reports roughly **1.3x** end-to-end speedup over the vanilla dLLM, rising to up to **40.2x** when combined with existing acceleration methods, while maintaining comparable task accuracy. Token generation speed improves further on long-context inference (HumanEval).'
    comparison: 'Stacking LaCache on top of existing accelerators like parallel decoding or DPad reduces latency further beyond what either baseline achieves alone, rather than competing with step-reduction methods, LaCache occupies a multiplicative position on top of them. Because FP8 quantization is applied only to the FFN, not the K/V cache, it does not target the same tensor as methods that compress the K/V cache itself.'
    significance: 'From an efficient-AI standpoint, this paper shows there is still a free lunch left in dLLM acceleration, lossless redundancy elimination that costs nothing in quality, and the 1.3x from that alone becomes a pure bonus underneath whatever lossy compression is layered on top. Its orthogonality to step-count-reduction methods is practically useful for real serving-stack design.'
    limitations: "Which dLLM backbones this generalizes to (SDAR, TraDo, and the broader block-diffusion family) and whether LSM's core invariance assumption stays safe when a committed token is later remasked or revoked are not stated at the abstract level reviewed here (reviewer judgment)."
    future_work: "No explicit future-work items are identifiable within the abstract-level scope this review draws from, the full paper's conclusion would need to be checked."
    resources: 'No public code repository was found, no public release verified.'
thread:
  ko: |-
    dLLM 가속 연구는 지금까지 대체로 두 갈래로 갈라져 있었다. 디노이징 스텝 수 자체를 줄이는 계열(병렬 디코딩, DPad, few-step distillation)과, 캐시나 표현을 압축해 스텝당 비용을 줄이는 계열(KV 캐시 양자화, 엔트로피 기반 갱신 스케줄)이다. LaCache는 이 두 갈래 어디에도 정확히 속하지 않는 세 번째 축을 연다. 스텝 수도, 캐시 압축률도 건드리지 않고, 그냥 이미 계산했는데 또 계산하고 있는 순수 낭비를 없앤다.

    개념적 전환은 무손실이라는 제약을 진지하게 받아들인 데 있다. 블록-확산 dLLM에서 한 블록을 생성하는 동안 그 블록 밖의 토큰들은 정의상 값이 바뀌지 않는다. 이는 근사가 아니라 아키텍처가 보장하는 사실이다. LaCache는 이 사실을 그대로 캐싱 정책으로 옮겼을 뿐이다. 그 위에 얹은 FP8 FFN 양자화만 손실이 있는 선택적 추가이며, 두 층을 분리해 둔 덕분에 손실 압축을 얼마나 공격적으로 켤지는 서빙 환경마다 독립적으로 고를 수 있다.

    이 논문이 여는 질문은 무손실 캐싱의 한계가 어디까지인가이다. 지금은 블록 경계 안에서의 불변성만 활용하지만, self-correction이나 revocation처럼 확정됐던 토큰이 나중에 바뀌는 경우까지 캐시 무효화 없이 안전하게 다룰 수 있는지는 아직 열려 있다. dLLM 가속 스택이 앞으로 스텝-감소, 캐시-압축, 순수 재계산-제거라는 세 층을 모두 쌓아 올리는 방향으로 수렴한다면, LaCache는 그 세 번째 층의 첫 이름 있는 사례가 된다.
  en: |-
    dLLM acceleration research has largely split into two lineages so far: methods that reduce the number of denoising steps outright (parallel decoding, DPad, few-step distillation), and methods that compress the cache or representation to cut the cost of each step (KV-cache quantization, entropy-gated refresh scheduling). LaCache doesn't quite belong to either lineage, it opens a third axis that touches neither step count nor cache compression ratio, and simply removes the pure waste of computing something that was already computed.

    The conceptual shift is taking losslessness seriously as a constraint. During generation of one block in a block-diffusion dLLM, the tokens outside that block are, by construction, guaranteed not to change value, this isn't an approximation, it's an architectural fact. LaCache simply turns that fact directly into a caching policy. The FP8 FFN quantization layered on top is the only lossy, optional addition, and keeping the two layers separate means a deployment can dial how aggressively to turn on lossy compression independently of the lossless layer underneath.

    The question this paper leaves open is how far lossless caching can be pushed, right now it exploits only within-block invariance, and it remains untested whether cases where a finalized token later changes (via self-correction or revocation) can be handled just as safely without cache invalidation. If dLLM acceleration stacks eventually converge on stacking all three layers, step reduction, cache compression, and pure redundancy elimination, LaCache stands as the first named instance of that third layer.
sparks:
  - ko: '저자들이 명시하지 않았지만 자연스럽게 따라오는 질문 — 블록 안 토큰이 self-correction이나 remasking으로 나중에 다시 바뀌는 경우, LSM의 캐시된 임베딩/RoPE/attention 상태를 어떻게 무효화해야 정확성을 유지할 수 있을까?'
    en: "A question the authors don't state but that follows naturally: when a token inside a block later changes via self-correction or remasking, what invalidation rule keeps LSM's cached embedding/RoPE/attention states correct?"
  - ko: 'FP8 FFN 양자화가 스텝-의존적 활성화 분포에 맞춰 그룹별로 조정된다는 점에서, 같은 스텝-의존적 그래뉼래러티 아이디어를 K/V 캐시 양자화에도 적용하면 어떤 효과가 있을지는 이 논문의 범위 밖이다.'
    en: "The paper's step-dependent per-group granularity for FP8 FFN quantization is scoped to FFN only, whether the same step-dependent-granularity idea would help if applied to K/V cache quantization instead is outside this paper's scope."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
