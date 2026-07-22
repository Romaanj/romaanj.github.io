---
title: "Surprise Forcing: What to Remember, When to Skip in Long Video Generation"
arxivId: "2607.18436"
authors: "Shuwei Shi, Zhen Li, Muyao Niu, Chuanhao Li, Bo Zheng, Kaipeng Zhang, Yinqiang Zheng"
date: 2026-07-22
tags: ["video-diffusion", "kv-cache", "streaming"]
topic: 'kv-cache'
summary: "Surprise Forcing is a training-free framework for streaming autoregressive video diffusion that re-admits evicted frames into a rolling KV cache under a feedback-controlled surprise budget, and separately skips denoising steps for chunks judged easy, improving long-horizon consistency without sacrificing real-time streaming."
summary_ko: "Surprise Forcing는 스트리밍 자기회귀 비디오 확산에서 롤링 KV 캐시가 놓친 프레임을 피드백 제어 서프라이즈 예산 아래 다시 불러들이고, 쉬운 청크는 디노이징 스텝을 건너뛰는 학습-불필요 프레임워크로, 실시간 스트리밍을 유지하면서 장기 일관성을 개선한다."
links: ["coda-nmp-video-diffusion"]
resources:
  - label: 'arXiv'
    url: 'https://arxiv.org/abs/2607.18436'
figures:
  - src: /figures/surprise-forcing/fig1.png
    caption: "The framework of Surprise Forcing: a Surprise-Gated Memory Bank re-admits evicted frames under a feedback-controlled budget, and Surprise-Aware Denoising skips steps for low-difficulty chunks."
    caption_ko: "Surprise Forcing의 프레임워크: Surprise-Gated Memory Bank가 피드백 제어 예산 아래 제거된 프레임을 다시 불러들이고, Surprise-Aware Denoising이 난이도가 낮은 청크의 디노이징 스텝을 건너뛴다."
    credit: "Figure 2 from arXiv:2607.18436 — authors' figure"
analysis:
  ko:
    background: '스트리밍 자기회귀 비디오 확산 모델은 이전에 생성한 프레임들을 롤링 KV 캐시로 유지하면서 매 청크(chunk)를 순차적으로 생성한다. 메모리와 계산량을 감당하기 위해 캐시는 오래된 프레임을 밀어내는 슬라이딩 윈도우 형태를 취한다.'
    problem: '롤링 캐시는 최신 프레임만 남기고 먼 과거의 시각적 근거를 잊어버리는데, 그 근거가 이후에도 여전히 중요할 수 있다. 동시에 모든 청크는 실제 난이도와 무관하게 동일한 수의 디노이징 스텝을 받는다 — 쉬운 청크에도 어려운 청크와 같은 계산을 쏟아붓는 낭비다.'
    prior_limits: '기존 방법들은 시간 순서(가장 오래된 것부터 제거)나 현재 시점과의 관련성만으로 프레임을 제거해왔다. 두 기준 모두 "얼마나 예상 밖(놀라운)인가"라는, 장기적으로 중요할 수 있는 신호를 반영하지 못한다.'
    goal: '추가 학습 없이, 제거된 프레임 중 정말 중요한 것만 선택적으로 다시 불러들이는 기억 메커니즘과, 청크별 난이도에 맞춰 디노이징 계산량을 조절하는 메커니즘을 함께 설계하는 것이 목표다.'
    method: 'Surprise-Gated Memory Bank는 제거된 프레임을 요약해 두었다가, 전역 편차(global deviation)와 최근접 이웃 참신성(nearest-neighbor novelty)으로 측정한 서프라이즈 점수가 높은 프레임만 피드백 제어 예산 안에서 다시 받아들인다. Surprise-Aware Denoising은 첫 디노이징 패스 이후 인접 프레임 간 유사도로 청크 난이도를 추정해, 쉬운 청크는 중간 디노이징 스텝을 건너뛴다. 마치 회의록을 쓸 때 모든 발언을 다 받아적는 대신, 예상 밖의 발언만 따로 표시해뒀다가 나중에 필요하면 다시 꺼내 보는 것과 비슷하다.'
    key_idea: '핵심은 "제거"와 "망각"을 분리한 것이다 — 프레임은 캐시에서 물리적으로는 밀려나지만 서프라이즈 요약이 남아 있어 필요하면 되돌아올 수 있다. 난이도 축에서도 비슷한 원리가 반복된다: 균일한 스텝 수 대신, 실제로 어려운 부분에만 계산을 집중한다.'
    validation: 'VBench, VBench-Long, VBench-2.0 세 벤치마크에서 장기 일관성과 화질을 정량 평가하고, 실시간 스트리밍 유지 여부를 확인한다. 정성적 비교로 피사체 일관성과 움직임의 매끄러움을 다른 방법들과 나란히 비교한다.'
    results: '논문은 세 VBench 계열 벤치마크 전반에서 장기 일관성과 화질이 개선되면서도 실시간 스트리밍 성능이 유지된다고 보고한다. 이 리뷰가 참조한 초록 수준에서는 구체적인 수치(예: 압축률, FPS, VBench 세부 점수)가 확인되지 않는다 — 논문 내 명시 없음.'
    comparison: '시간 순서나 단순 관련성 기반의 기존 제거 전략과 비교해, 서프라이즈 기반 재수용이 장기 일관성에서 우위를 보인다고 보고된다. 정성적 비교에서 피사체 일관성과 움직임 연속성이 다른 방법 대비 강하다고 주장한다.'
    significance: '효율적 비디오 생성 관점에서, 이 논문은 KV 캐시 압축에서 "제거된 정보는 영구히 사라진다"는 암묵적 전제에 의문을 던진다 — 압축과 복원 가능성을 분리하면, 고정 예산 안에서도 정말 중요한 과거 정보를 지킬 수 있다는 것이다. 난이도-적응 디노이징 역시 균일 스텝 배분이라는 또 다른 흔한 전제를 겨냥한다.'
    limitations: '서프라이즈 점수 자체가 얼마나 신뢰할 수 있는 신호인지(즉 "예상 밖"이 실제로 "중요함"과 얼마나 일치하는지), 그리고 재수용 예산이 소진되었을 때의 동작은 이 리뷰가 참조한 초록 수준에서 명시되지 않는다(리뷰어 판단).'
    future_work: '논문 내 명시된 future-work 항목은 이 리뷰가 참조한 초록 범위에서는 확인되지 않는다.'
    resources: '공개 코드 저장소는 확인되지 않았다 — 공개 링크 확인 안 됨.'
  en:
    background: 'Streaming autoregressive video diffusion models generate chunks sequentially while keeping a rolling KV cache of previously generated frames. To keep memory and compute bounded, the cache takes the form of a sliding window that evicts older frames.'
    problem: 'A rolling cache forgets distant visual evidence once it slides out of the window, even when that evidence remains important later. At the same time, every chunk receives the same number of denoising passes regardless of how difficult it actually is, wasting compute on chunks that were easy to begin with.'
    prior_limits: 'Prior eviction strategies rely purely on temporal order (oldest goes first) or current relevance. Neither criterion captures how surprising, and therefore potentially important later, a piece of evidence is.'
    goal: 'Without any additional training, design a memory mechanism that selectively re-admits only the evicted frames that matter, alongside a mechanism that adapts denoising compute to each chunk''s actual difficulty.'
    method: "The Surprise-Gated Memory Bank summarizes evicted frames and re-admits only those with a high surprise score, measured by global deviation and nearest-neighbor novelty, within a feedback-controlled budget. Surprise-Aware Denoising estimates each chunk's difficulty from adjacent-frame similarity after an initial denoising pass and skips intermediate steps for easy chunks. It resembles taking meeting minutes: instead of writing down every single remark, you flag only the unexpected ones for later reference, and pull them back up if they turn out to matter."
    key_idea: 'The core move is decoupling eviction from forgetting: a frame is physically pushed out of the cache, but its surprise summary survives, so it can come back if needed. A parallel logic applies on the difficulty axis: instead of a uniform step count, compute concentrates only where the content is actually hard.'
    validation: 'Long-horizon consistency and visual quality are evaluated quantitatively on VBench, VBench-Long, and VBench-2.0, alongside a check that real-time streaming is preserved. Qualitative comparisons assess subject consistency and motion smoothness side-by-side with other methods.'
    results: 'The paper reports improved long-horizon consistency and visual quality across all three VBench-family benchmarks while maintaining real-time streaming performance. Specific numbers (e.g. compression ratio, FPS, per-benchmark VBench scores) are not identifiable at the abstract-level scope this review draws from — not stated in the paper (from this review''s vantage point).'
    comparison: 'Against prior eviction strategies based purely on temporal order or relevance, surprise-gated re-admission is reported to win on long-horizon consistency. Qualitative comparisons claim stronger subject consistency and motion continuity than other methods.'
    significance: 'From an efficient-video-generation standpoint, this paper questions the implicit assumption in KV-cache compression that evicted information is gone for good — decoupling compression from recoverability lets a fixed budget still protect the past evidence that actually matters. The difficulty-adaptive denoising half targets a separate common assumption: uniform step allocation.'
    limitations: 'How reliable the surprise score itself is as a signal (how well "unexpected" tracks "important"), and what happens when the re-admission budget is exhausted, are not stated at the abstract-level scope this review draws from (reviewer judgment).'
    future_work: 'No explicit future-work items are identifiable within the abstract-level scope this review draws from.'
    resources: 'No public code repository was found, no public release verified.'
thread:
  ko: |-
    스트리밍 비디오 확산의 KV 캐시 관리는 지금까지 대체로 "무엇을 버릴 것인가"라는 질문으로 좁혀져 있었다. 시간 순서 기반 슬라이딩 윈도우가 가장 단순한 답이었고, 이후 연구들은 현재 프레임과의 관련성을 기준으로 조금 더 똑똑하게 버리는 쪽으로 발전해왔다. 그러나 두 접근 모두 한 번 버린 정보는 완전히 사라진다는 전제를 공유한다.

    Surprise Forcing의 전환은 그 전제 자체를 깬 데 있다 — 버림(eviction)과 잊음(forgetting)을 같은 것으로 취급하지 않고, 제거된 프레임의 요약을 남겨 서프라이즈 신호가 높으면 예산 안에서 되불러올 수 있게 만들었다. 이는 "무엇을 버릴 것인가"에서 "무엇을 되돌릴 수 있게 남겨둘 것인가"로 질문 자체를 바꾼 것이다. 같은 논문이 디노이징 스텝 배분에도 같은 논리를 적용해, 계산량 역시 고정된 것이 아니라 콘텐츠의 실제 난이도에 반응하는 자원으로 다룬다.

    이 논문이 여는 질문은 서프라이즈라는 신호가 얼마나 일반적인 척도로 쓰일 수 있는가이다 — 지금은 시각적 편차와 참신성으로 정의되지만, 같은 원리가 텍스트나 다른 시퀀스 생성 모델의 캐시 관리에도 이식될 수 있을지, 그리고 재수용 예산이 소진되는 장기 스트리밍 환경에서는 어떤 실패 양상을 보이는지는 아직 다뤄지지 않았다.
  en: |-
    KV-cache management for streaming video diffusion has mostly been framed as a narrow question: what to discard. A temporal-order sliding window was the simplest answer, and subsequent work refined it toward discarding based on relevance to the current frame. Both approaches share the same underlying assumption, though: once evicted, information is gone for good.

    Surprise Forcing's shift is breaking that assumption. Rather than treating eviction and forgetting as the same event, it keeps a summary of what was evicted, and lets high-surprise content come back within a budget. That reframes the question from "what to discard" to "what to keep recoverable." The same paper applies the same logic to denoising-step allocation, treating compute not as a fixed quantity but as a resource that should respond to how difficult the content actually is.

    The question this paper leaves open is how general the surprise signal can become — here it is defined via visual deviation and novelty, but whether the same principle could transfer to cache management in text or other sequence-generation models, and how the system degrades once the re-admission budget is exhausted in very long streaming sessions, are both left unexplored.
sparks:
  - ko: '서프라이즈 신호가 시각적 편차와 최근접 이웃 참신성으로만 정의되는데, 텍스트나 다른 모달리티의 시퀀스 생성 모델에서도 유사한 "예상 밖 정도" 신호로 캐시 재수용을 설계할 수 있을까?'
    en: "The surprise signal here is defined purely via visual deviation and nearest-neighbor novelty — could an analogous unexpectedness signal drive cache re-admission in text or other-modality sequence generation models?"
  - ko: '재수용 예산이 소진된 뒤 장기 스트리밍이 계속될 때 시스템이 어떻게 동작하는지는 논문에 나타나지 않는다 — 예산 고갈 시나리오에서의 성능 저하 양상은 저자들이 미래 과제로 명시하지 않았지만 자연스럽게 따라오는 질문이다.'
    en: "What happens once the re-admission budget is exhausted during a very long streaming session isn't shown in the paper — the degradation behavior under budget exhaustion is a natural follow-up question, though not one the authors flag as future work themselves."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
