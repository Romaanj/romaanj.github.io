---
title: "HeadCast: Casting Attention Heads for Efficient Autoregressive Video Generation"
arxivId: "2607.20125"
authors: "Jinliang Shen, Lianghao Su, Zheming Li, Kang He, Ziliang Lai, Yanbing Jiang, Chengru Song"
lab: "KlingAI Research"
date: 2026-07-23
tags: ["kv-cache", "video-diffusion", "eviction"]
topic: 'kv-cache'
summary: "HeadCast is a training-free KV-cache acceleration framework for autoregressive video diffusion that classifies each attention head into one of four stable behavioral archetypes (Sink, Dummy, Spatial, Global) and routes each to a disjoint storage/compute pathway, keeping the long-range Global heads at full attention to avoid the flickering that coarser eviction baselines cause."
summary_ko: "HeadCast는 자기회귀 비디오 확산을 위한 학습-불필요 KV 캐시 가속 프레임워크로, 각 어텐션 헤드를 Sink·Dummy·Spatial·Global 네 가지 안정적 행동 유형으로 분류해 서로 다른 저장/연산 경로로 라우팅하며, 장거리 의존성을 지닌 Global 헤드는 전체 어텐션으로 남겨 두어 거친 제거 기법이 유발하는 플리커링을 피한다."
links: ["self-gradient-forcing", "surprise-forcing"]
resources:
  - label: 'arXiv'
    url: 'https://arxiv.org/abs/2607.20125'
  - label: 'GitHub'
    url: 'https://github.com/sjlgaga/HeadCast'
figures:
  - src: /figures/headcast/fig1.png
    caption: "The four-phase HeadCast pipeline: full-context Warm-up, Online Classification, Heterogeneous Cache Management, and Head-Specific Attention."
    caption_ko: "HeadCast의 4단계 파이프라인 — 전체-컨텍스트 Warm-up, Online Classification, Heterogeneous Cache Management, Head-Specific Attention."
    credit: "Figure 2 from arXiv:2607.20125 — authors' figure"
  - src: /figures/headcast/fig2.png
    caption: "The Spatial Path: each frame is partitioned into non-overlapping grids, and a query attends only to KV tokens in its own grid across history, decomposing dense attention into independent sub-matrices whose savings grow with resolution."
    caption_ko: "Spatial Path — 각 프레임을 겹치지 않는 격자로 나누고, 쿼리는 히스토리 전체에서 자기 격자 안의 KV 토큰에만 주의를 기울여 밀집 어텐션을 독립적인 부분행렬들로 분해한다. 절감 폭은 해상도가 커질수록 늘어난다."
    credit: "Figure 3 from arXiv:2607.20125 — authors' figure"
analysis:
  ko:
    background: "자기회귀(AR) 비디오 확산 모델은 긴 영상을 블록 단위로 생성하며 KV 캐시로 과거 프레임 상태를 재사용한다. 하지만 캐시는 생성이 진행될수록 계속 커지고, 어텐션 비용은 캐시 길이에 대해 이차적으로 증가해 결국 추론의 병목이 된다 — 특히 프레임당 토큰 수가 많은 고해상도에서 두드러진다."
    problem: "캐시를 줄이는 기존 방법들은 거친 휴리스틱으로 오래된 컨텍스트를 제거하거나(**프레임 간 플리커링** 유발), 재학습이 필요하다. 저자들은 기존 제거 베이스라인(**Dummy Forcing**)이 사실상 모든 헤드를 지역 윈도우만 보는 Dummy 유형으로 취급한다는 점을 지목한다 — 일부 헤드가 실제로는 장거리 컨텍스트에 의존하는데 이를 함께 제거해버려 구조가 깨진다는 것이다."
    prior_limits: "기존 제거 전략은 헤드를 구분하지 않는 **균일한** 정책을 적용하거나, 재학습 기반 희소 어텐션처럼 배포 비용이 큰 방식에 의존해왔다. 어떤 헤드가 실제로 장거리 정보를 필요로 하는지 구분하는 저비용 진단이 빠져 있었다."
    goal: "재학습 없이, 사전학습된 AR 모델의 각 어텐션 헤드가 실제로 어떤 컨텍스트를 필요로 하는지 값싸게 진단해, 헤드별로 서로 다른 저장/연산 경로를 부여하는 것이 목표다."
    method: "**HeadCast**는 워밍업 후 최대-노이즈 스텝에서 단 한 번 분류를 수행한다. 각 헤드의 어텐션 출력을 전체-히스토리 참조 출력과 코사인 유사도로 비교해, **Sink**(첫 시간 블록에만 주의), **Dummy**(현재 지역 블록만), **Spatial**(전체 히스토리이지만 쿼리 주변 그리드로 국한), **Global**(위 어디에도 맞지 않음 — 장거리 시간 일관성을 담당) 네 유형 중 하나로 배정한다. Spatial 판정에는 평균 코사인 유사도가 일부 경계 토큰의 실패를 가려버릴 수 있다는 점을 감안해, 5백분위수 코사인 유사도에서 MSE 페널티를 뺀 보수적 점수를 쓴다. 마치 회의 참석자 각각의 평소 발언 패턴을 미리 파악해 두었다가, 회의록을 압축할 때 각자에게 맞는 요약 방식을 다르게 적용하는 것과 비슷하다."
    key_idea: "핵심은 제거를 **이진(유지/삭제)** 결정이 아니라 헤드별로 **네 갈래의 서로 다른 경로**로 다루는 것이다 — 특히 장거리 일관성을 담당하는 Global 헤드를 명시적으로 보호함으로써, 균일 제거가 만드는 플리커링을 원천적으로 피한다. 해상도가 커질수록 Spatial 경로의 절감폭이 자동으로 커진다는 점도 설계상의 장점이다."
    validation: "4개의 최신 AR 비디오 백본에서 480P·720P·1080P 해상도, VBench(품질·의미·전체·역동성), 전체 어텐션 대비 PSNR/LPIPS, 블록 경계 불연속성 지표, 사용자 연구, 분류 임계값·시점 소거 실험으로 평가한다."
    results: "480P에서는 캐시가 아직 작아 두 방법 모두 속도 향상이 크지 않지만(HeadCast 1.02–1.08배), 고해상도에서 격차가 벌어진다 — LongLive(30초) 백본에서 720P/1080P 각각 **1.62배/1.95배** 속도 향상, 정상상태 KV 캐시는 전체 어텐션 대비 약 **33% 축소**. 프레임 충실도에서 HeadCast는 전체-어텐션 대비 22–26dB PSNR(Dummy Forcing 16–19dB)·LPIPS 0.045–0.068(Dummy Forcing 0.15–0.22, 3–4배 격차)을 기록하며, 블록 경계 불연속성 지표에서 Dummy Forcing이 최대 36% 악화되는 반면 HeadCast는 전체 어텐션과 동등하다."
    comparison: "동일한 4개 백본에서 전체 어텐션(상한)과 기존 거친 제거 베이스라인(**Dummy Forcing**)을 함께 비교한다. VBench 총점은 HeadCast가 전체 어텐션과 0.15점 이내로 근접하는 반면 Dummy Forcing은 최대 0.7점까지 벌어진다 — 두 방법의 속도 차이보다 화질·일관성 격차가 더 뚜렷하게 갈린다."
    significance: "효율적 비디오 생성 관점에서, 이 논문은 헤드가 모두 똑같이 히스토리 전체를 봐야 한다는 균일-정책 전제에 의문을 던진다 — 헤드가 이미 안정적으로 이질적(heterogeneous)이라면, 값싼 일회성 진단만으로 재학습 없이 그 이질성을 활용할 수 있다는 것이다."
    limitations: "네 아키타입의 결정 임계값이 얼마나 좁은 범위에서만 잘 작동하는지, 즉 사실상의 튜닝 비용이 어느 정도인지는 논문의 임계값 소거 실험(Fig. 5)에 나타나지만 이 리뷰가 읽은 범위에서 구체적 민감도 수치까지 확정하지는 않는다(리뷰어 판단). 분류가 최대-노이즈 스텝 한 번으로 고정된다는 가정이 이 논문에서 시험한 것보다 훨씬 긴 롤아웃이나 콘텐츠 분포 변화에서도 유지되는지는 논문 내 명시 없음."
    future_work: "이 리뷰가 읽은 범위(초록·서론·방법·주요 결과 섹션)에서 저자들이 명시한 future-work 항목은 확인되지 않는다 — 논문 내 명시 없음."
    resources: "공개 GitHub 저장소(https://github.com/sjlgaga/HeadCast)가 논문 초록에 명시되어 있고 접속이 확인되었다."
  en:
    background: "Autoregressive (AR) video diffusion models generate video block-by-block, reusing past frame state via a KV cache. As generation proceeds the cache grows without bound and attention cost scales quadratically with cache length, eventually dominating inference -- especially at high resolution, where each frame contributes many tokens."
    problem: "Existing cache-reduction methods either evict old context with coarse heuristics (causing **inter-frame flickering**) or require retraining. The authors pinpoint that the existing eviction baseline (**Dummy Forcing**) effectively treats nearly every head as a local-window-only Dummy type -- evicting long-range context that some heads actually depend on, breaking structure those heads were carrying."
    prior_limits: "Prior eviction strategies apply a **uniform** policy across heads, or rely on retraining-based sparse attention with high deployment cost. What was missing was a cheap diagnostic for which heads actually need long-range information and which don't."
    goal: "Without any retraining, cheaply diagnose what context each attention head of a pretrained AR model actually needs, and give each head a different storage/compute pathway accordingly."
    method: "**HeadCast** performs a single one-time classification after warm-up, at the maximum-noise denoising step. Each head's attention output is compared to a full-history reference output via cosine similarity, sorting it into one of four archetypes: **Sink** (attends only to the first temporal block), **Dummy** (only the current local block), **Spatial** (all history, but confined to a local grid around each query token), or **Global** (fits none of the above -- the heads carrying long-range temporal consistency). The Spatial decision uses a conservative score (5th-percentile cosine similarity minus an MSE penalty) rather than a plain average, since average similarity can stay high even when a head fails at a few boundary tokens. It is a bit like profiling each meeting participant's usual speaking pattern in advance, then applying a different note-taking shorthand to each person when compressing the minutes."
    key_idea: "The core move is treating cache reduction as a **four-way routing** decision per head rather than a binary keep/evict choice -- explicitly protecting the Global heads that carry long-range consistency avoids the flickering that uniform eviction causes. A useful side effect of the design is that the Spatial pathway's savings automatically grow with resolution."
    validation: "Evaluated on four state-of-the-art AR video backbones at 480P/720P/1080P, using VBench (Quality, Semantic, Total, Dynamic Degree), PSNR/LPIPS against full-attention output, a block-boundary discontinuity metric, a user study, and ablations over classification thresholds and the classification timestep."
    results: "At 480P the cache is still small so speedups are modest for both methods (HeadCast 1.02-1.08x), but the gap widens at high resolution -- on the LongLive (30s) backbone, HeadCast reaches **1.62x/1.95x** at 720P/1080P, with steady-state KV-cache size roughly **33% smaller** than full attention. On frame fidelity, HeadCast reaches 22-26 dB PSNR (vs. Dummy Forcing's 16-19 dB) and 0.045-0.068 LPIPS (vs. 0.15-0.22, a 3-4x gap) against each model's full-attention output; a block-boundary discontinuity metric shows Dummy Forcing degrading by up to 36% while HeadCast matches full attention."
    comparison: "Compared against both full attention (the ceiling) and the existing coarse eviction baseline (**Dummy Forcing**) across the same four backbones. HeadCast stays within 0.15 VBench-Total points of full attention while Dummy Forcing loses up to 0.7 points -- the fidelity/consistency gap between the two methods is far sharper than their speed gap."
    significance: "From an efficient-video-generation standpoint, this paper questions the uniform-policy assumption that every head needs to see the same amount of history -- if heads are already stably heterogeneous, a cheap one-time diagnostic is enough to exploit that heterogeneity without any retraining."
    limitations: "How narrow the good-operating-range is for the four archetype thresholds -- i.e. the effective hidden tuning cost -- is shown in the paper's threshold ablation (Fig. 5), but this review's read doesn't pin down specific sensitivity numbers (reviewer judgment). Whether the one-shot max-noise-step classification assumption holds at much longer rollouts or under content-distribution shift than tested here is not stated in the paper."
    future_work: "No explicit future-work items are identifiable within the scope this review draws from (abstract, method, and main-results sections) -- not stated in the paper."
    resources: "A public GitHub repository (https://github.com/sjlgaga/HeadCast) is stated in the paper's abstract and was confirmed reachable."
thread:
  ko: |-
    AR 비디오 확산의 KV 캐시 절감 계열은 대체로 두 축으로 갈라져 왔다: 시간/관련성 기반 제거(슬라이딩 윈도우류)와, 최근에는 서프라이즈 신호로 제거를 되돌리는 재수용 방식(Surprise Forcing, [[surprise-forcing]]). 두 계열 모두 무엇을 저장할 것인가를 **토큰/프레임** 단위로 결정한다는 공통점이 있다.

    이 논문은 단위 자체를 바꾼다 — 결정을 프레임이 아니라 **헤드** 단위로 내린다. 모든 프레임에 걸쳐 균일하게 적용되는 하나의 제거 정책 대신, 헤드마다 안정적으로 다른 행동 패턴(Sink/Dummy/Spatial/Global)이 있다는 것을 먼저 진단하고, 그 진단에 따라 서로 다른 저장 전략을 부여한다. 이는 제거냐 보존이냐의 질문을, 이미 어떤 헤드는 지역적이고 어떤 헤드는 전역적이라는 사실을 인정하는 쪽으로 재구성한 것이다.

    이 흐름이 여는 질문은 두 가지다. 하나는 헤드 단위 라우팅과 프레임/토큰 단위 재수용(Surprise Forcing류)을 같은 시스템에서 결합할 수 있는가 — 서로 직교하는 축처럼 보이기 때문이다. 다른 하나는 이 헤드 이질성이 비디오 확산에 국한된 현상인지, 아니면 텍스트나 다른 모달리티의 자기회귀 생성에서도 같은 방식으로 진단하고 활용할 수 있는 더 일반적인 구조인지다.
  en: |-
    KV-cache reduction for AR video diffusion has mostly split into two lineages: temporal/relevance-based eviction (sliding-window style), and, more recently, surprise-gated re-admission that partially reverses eviction (Surprise Forcing, [[surprise-forcing]]). Both lineages share a common unit of decision: what to keep is decided per token/frame.

    This paper changes the unit itself -- the decision is made per head, not per frame. Instead of one eviction policy applied uniformly across every frame, it first diagnoses that heads have stably different behavior patterns (Sink/Dummy/Spatial/Global), then assigns a different storage strategy per pattern. That reframes evict-or-keep into acknowledging a fact that was already true: some heads are inherently local and some are inherently global.

    This opens two questions. One is whether head-level routing and frame/token-level re-admission (Surprise-Forcing-style) can be combined in the same system, since they look like orthogonal axes. The other is whether this head heterogeneity is specific to video diffusion, or a more general structure that could be diagnosed and exploited the same way in text or other-modality autoregressive generation.
sparks:
  - ko: "논문은 헤드 분류가 최대-노이즈 스텝 한 번으로 고정된다고 가정한다 — 훨씬 긴 롤아웃이나 장면이 크게 바뀌는 콘텐츠에서 헤드 아키타입이 실제로 안정적으로 유지되는지 직접 시험해보면 이 가정의 한계를 드러낼 수 있을 것이다."
    en: "The paper assumes head classification stays fixed after a single max-noise-step pass -- directly testing whether head archetypes actually remain stable over much longer rollouts or under large scene/content shifts would probe the limits of that assumption."
  - ko: "네 아키타입 사이의 임계값 민감도 소거 실험은 있지만, 더 거친 분류로도 대부분의 이득을 회수할 수 있는지는 별도로 다뤄지지 않는다 — 아키타입 수 자체를 바꿔가며 성능-복잡도 트레이드오프를 그려보는 것은 자연스러운 후속 질문이다."
    en: "There's a threshold-sensitivity ablation across the four archetypes, but whether a coarser classification recovers most of the benefit isn't separately addressed -- sweeping the number of archetypes itself to map the performance-complexity trade-off is a natural follow-up."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
