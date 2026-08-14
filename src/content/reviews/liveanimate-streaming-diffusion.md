---
title: "LiveAnimate: Stable Long-Form Streaming Human Animation in Real-Time"
arxivId: "2608.11745"
date: 2026-08-15
tags: ["kv-cache", "video-diffusion", "streaming", "sink-attention"]
topic: 'kv-cache'
summary: "A block-causal video diffusion model with a retrieval-augmented sink-attention KV-cache policy sustains real-time, constant-memory streaming human animation for minutes, not seconds."
summary_ko: "블록-코잘 비디오 디퓨전 모델에 검색 기반 싱크-어텐션 KV 캐시 정책을 결합해, 수 분 길이의 스트리밍 휴먼 애니메이션을 실시간·일정 메모리로 유지한다."
links: ["self-gradient-forcing", "surprise-forcing", "duo-attention", "sana-video2"]
resources:
  - label: "arXiv abstract"
    url: "https://arxiv.org/abs/2608.11745"
  - label: "arXiv PDF"
    url: "https://arxiv.org/pdf/2608.11745"
figures:
  - src: /figures/liveanimate-streaming-diffusion/fig1.png
    caption: "LiveAnimate overview: each video block is denoised in 3 steps then followed by a clean KV update; PR-Sink augments a 3-block rolling window with the first generated block and a pose-matched historical block from a compact memory bank."
    caption_ko: "LiveAnimate 개요: 각 비디오 블록은 3단계로 디노이징된 뒤 클린 KV 업데이트가 뒤따른다. PR-Sink는 3블록 롤링 윈도우에 최초 생성 블록과, 소형 메모리 뱅크에서 포즈로 매칭한 과거 블록을 추가한다."
    credit: "Figure 2 from arXiv:2608.11745 — authors' figure"
analysis:  # per key: 2-4 sentences — first = one crisp information-dense claim; add an apt analogy where it genuinely clarifies
  ko:
    background: "포즈 기반 휴먼 애니메이션은 단일 레퍼런스 이미지와 구동 포즈(스켈레톤) 스트림을 받아, 그 인물이 구동 동작을 수행하는 영상을 만든다 — 가상 아바타, 텔레프레즌스, 라이브 스트리밍에 쓰인다. 디퓨전 트랜스포머는 이제 매우 높은 품질의 애니메이션을 만들지만, 표준 디퓨전 샘플링은 클립 전체를 한 번에 디노이징하고 여러 단계를 필요로 하므로 본질적으로 오프라인 방식이다."
    problem: "라이브 스트리밍·텔레프레즌스 같은 인터랙티브 용도는 수백 밀리초 안에 반응을 시작해 수 분간 안정적으로 계속 돌아가는 애니메이션을 필요로 하지만, 논문이 비교 대상으로 삼는 디퓨전 기반 시스템들은 클립 하나에 수 분에서 수 시간이 걸린다. 논문은 핵심 난제를 보통 서로 상충하는 두 가지를 동시에 만족시켜야 하는 문제로 규정한다 — 실시간·일정 비용의 스트리밍 생성, 그리고 장시간 스트림에서도 드리프트하거나 레퍼런스 정체성을 잃지 않는 안정적인 장시간 품질."
    prior_limits: "이 사이트가 다뤄온 Self-Forcing/CausVid 계열의 기존 실시간 스트리밍 비디오 디퓨전 접근은 양방향 디퓨전 트랜스포머를 블록-코잘 자기회귀 생성기로 바꾸고 단순한 롤링 KV 캐시 윈도우를 쓰는데, 이는 최근 맥락은 유지하지만 윈도우 밖의 모든 것은 잊는다. 포즈 기반 애니메이션에서는 이것이 특히 문제인데, 간격을 두고 포즈가 다시 나타날 때(예: 피사체가 비슷한 자세로 돌아올 때) 그 포즈의 정체성·외형 맥락이 이미 윈도우 밖으로 밀려나 있을 수 있어 장시간 스트림에서 품질과 정체성이 드리프트하기 때문이다."
    goal: "10억 파라미터급 비디오 디퓨전 모델에서 실시간 스트리밍 생성(목표: 수십 FPS)을 수 분 단위의 긴 스트림에서 유지하되, 스트림이 얼마나 오래 실행됐는지와 무관하게 메모리와 블록당 지연시간을 일정하게 유지하고, 시간이 지나도 피사체 정체성을 잃거나 품질이 저하되지 않아야 한다."
    method: "LiveAnimate는 세 부분으로 구성된다. (1) Reference-Anchored Teacher-Forcing Adaptation이 사전학습된 양방향 140억 파라미터 비디오 DiT를 블록-코잘 자기회귀 생성기로 바꾼다. (2) Block-wise Self-Forcing Distillation이 샘플링 예산을 블록당 **3단계**의 디노이징으로 줄인다. (3) Pose-Retrieval Sink Attention(PR-Sink)은 최초 생성 블록(정체성·외형 레퍼런스)을 영구히 고정하는 Static Sink, 현재 구동 포즈가 과거 포즈와 일치할 때 소형 메모리 뱅크에서 포즈로 검색해 온 과거 블록 하나를 담는 Dynamic Sink, 그리고 최근 맥락을 위한 3슬롯 롤링 윈도우를 결합한 유한 KV 캐시 정책이다 — 여기에 처리량을 위한 Ulysses 시퀀스 병렬화와 연산자 퓨전이 더해진다."
    key_idea: "차별점은 Dynamic Sink에 있다 — 단순 롤링 윈도우처럼 위치로만 잊는(가장 오래된 블록부터 축출) 캐시 대신, PR-Sink는 지금 생성 중인 포즈가 과거의 특정 포즈와 일치한다는 이유로 그 특정 과거 블록을 되돌아가 검색해 올 수 있다. 이는 마지막 몇 페이지만 다시 볼 수 있는 일기장보다는, 특정 페이지로 되돌아갈 수 있는 사진 앨범에 가깝다 — 피사체가 10초 전에 취했던 포즈를 다시 취하면, 캐시는 그 순간의 맥락을 처음부터 다시 만들어내거나 완전히 잃는 대신 검색해 온다."
    validation: "3분 길이 스트리밍 벤치마크에서 첫 30초부터 마지막 1분까지 지각 품질과 피사체 정체성을 추적해 평가하며, 동일 지표로 오프라인 디퓨전 베이스라인과 비교한다. 처리량은 Ulysses 시퀀스 병렬화와 연산자 퓨전을 켠 2×H100 GPU에서의 end-to-end FPS로 측정한다."
    results: "LiveAnimate는 H100 2장에서 **19.63 FPS**의 스트리밍 추론을 유지한다. 3분 벤치마크에서 첫 30초부터 마지막 1분까지 지각 품질과 정체성이 거의 일정하게 유지되는 반면, 비교 대상 오프라인 베이스라인들은 같은 구간에서 상당히 저하되거나 비슷한 클립을 만드는 데 수 시간의 오프라인 연산이 필요하다. 메모리와 블록당 지연시간은 스트림 길이와 무관하게 일정한데, 이는 PR-Sink의 유한 캐시 설계가 직접 가져오는 결과다."
    comparison: "이 사이트가 다뤄온 Self-Forcing/CausVid 계열의 단순 롤링 KV 캐시 레시피와 비교하면, LiveAnimate의 기여는 정확히 캐시 정책 수준에 있다 — 새로운 증류나 어텐션 메커니즘을 제안하는 대신, 통상의 최신성 기반 롤링 윈도우 위에 콘텐츠/검색 기반 Dynamic Sink를 추가한다. 3단계 셀프포싱 증류 아이디어 자체는 이 계열의 다른 곳에서도 본 것과 같으며, 여기서는 포즈가 반복되기 때문에 검색 기반 캐시가 특히 도움이 되는 포즈 기반 애니메이션 설정에 적용된다."
    significance: "스트리밍/코잘 비디오 디퓨전을 위한 KV 캐시 설계를 추적하는 입장에서, PR-Sink는 순수하게 위치 기반이 아니라 콘텐츠 조건부인 캐시 축출 정책의 구체적 사례다 — 이 분야의 대부분 축출 정책(최신성 윈도우, 중요도 점수)은 여전히 미래 쿼리가 실제로 무엇을 필요로 할지를 무시하는 신호로 무엇을 유지할지 결정한다. 보존 여부를 '입력 신호(포즈)가 주기적이거나 반복적이어서 다시 필요해질 것인가'에 연결하는 것은, 조건 신호 자체가 반복되는 어떤 스트리밍 생성 설정(예: 주기적 동작, 반복되는 카메라 앵글, 루프가 많은 콘텐츠)에도 재사용 가능한 아이디어다."
    limitations: "논문은 LiveAnimate가 현재 3단계 디노이징으로 **480×480** 해상도로 생성하며, 이것이 시각적 충실도와 해상도를 제한한다고 밝힌다. 또한 다인물 장면이나 큰 카메라 움직임은 지원하지 않는다고 명시한다."
    future_work: "논문은 LiveAnimate를 고해상도 스트리밍 생성, 다인물 애니메이션, 카메라가 움직이는 장면으로 확장하는 것을 향후 과제로 남겨둔다고 밝힌다."
    resources: "공개 링크 확인 안 됨. 이번 리뷰에서 읽은 범위에서는 논문에 코드·모델·프로젝트 페이지 공개에 대한 언급이 없다."
  en:
    background: "Pose-driven human animation turns a single reference image plus a stream of driving poses (a skeleton sequence) into video of that person performing the driving motion — used for virtual avatars, telepresence, and live streaming. Diffusion transformers now produce very high quality animation, but standard diffusion sampling denoises a whole clip at once and needs many steps, which is fundamentally offline."
    problem: "Interactive use (live streaming, telepresence) needs animation that starts responding within a fraction of a second and keeps running stably for minutes, but the diffusion-based systems the paper compares against take minutes to hours per clip. The paper frames the core difficulty as needing two things simultaneously that usually trade off against each other: real-time, constant-cost streaming generation, and stable long-form quality that does not drift or lose the reference identity over an extended stream."
    prior_limits: "Prior real-time approaches to streaming video diffusion (the Self-Forcing/CausVid line this site has covered) convert bidirectional diffusion transformers into block-causal autoregressive generators with a plain rolling KV-cache window, which keeps recent context but forgets everything outside the window. For pose-driven animation specifically that is a problem when a pose recurs after a gap — e.g. the subject returns to a similar stance — because the identity/appearance context for that pose may already have scrolled out of the window, and quality/identity drifts over long streams as a result."
    goal: "Sustain real-time streaming generation (target: tens of FPS) for a billion-scale video diffusion model over long streams (minutes, not seconds), while keeping memory and per-block latency constant regardless of how long the stream has been running, and without losing subject identity or degrading quality over time."
    method: "LiveAnimate has three parts. (1) Reference-Anchored Teacher-Forcing Adaptation converts a pretrained bidirectional 14B video DiT into a block-causal autoregressive generator. (2) Block-wise Self-Forcing Distillation reduces the sampling budget to **3 denoising steps** per block. (3) Pose-Retrieval Sink Attention (PR-Sink) is a bounded KV-cache policy combining a Static Sink that permanently anchors the very first generated block (the identity/appearance reference), a Dynamic Sink holding one pose-retrieved historical block pulled from a compact memory bank when the current driving pose matches a past one, and a 3-slot rolling window for recent context — plus Ulysses sequence parallelism and operator fusion for throughput."
    key_idea: "The Dynamic Sink is the distinguishing move: instead of a cache that only ever forgets by position (oldest block evicted first, as in a plain rolling window), PR-Sink can reach back and retrieve a specific historical block because its pose matches the pose being generated right now. It behaves less like a diary you can only read the last few pages of, and more like a photo album you can flip back to a specific page in — when the subject strikes a pose seen ten seconds ago, the cache retrieves that moment's context instead of having to re-derive it from scratch or lose it entirely."
    validation: "Evaluated on a three-minute streaming benchmark, tracking perceptual quality and subject identity from the first 30 seconds through the final minute, and measured against offline diffusion baselines under the same metrics. Throughput is measured as end-to-end FPS on 2xH100 GPUs with Ulysses sequence parallelism and operator fusion enabled."
    results: "LiveAnimate sustains **19.63 FPS** streaming inference on two H100 GPUs. On the three-minute benchmark it maintains nearly constant perceptual quality and identity from the first 30 seconds to the final minute, while the offline baselines it compares against degrade substantially over the same window or require hours of offline computation to produce a comparable clip. Memory and per-block latency stay constant regardless of stream duration, which is the direct payoff of PR-Sink's bounded cache design."
    comparison: "Relative to the plain rolling-KV-cache recipe this site has covered in the Self-Forcing/CausVid lineage, LiveAnimate's contribution is specifically at the cache-policy level: it adds a content/retrieval-keyed Dynamic Sink on top of the usual recency-keyed rolling window, rather than proposing a new distillation or attention mechanism. It is the same three-step self-forcing distillation idea seen elsewhere in this lineage, applied here to the pose-driven-animation setting where a retrieval-based cache specifically helps because poses recur."
    significance: "For anyone tracking KV-cache design for streaming/causal video diffusion, PR-Sink is a concrete example of a cache eviction policy that is content-conditioned rather than purely positional — most eviction policies in this space (recency windows, saliency scores) still decide what to keep using signals that ignore what future queries will actually need. Tying retention to whether the input signal (pose) is periodic or recurring is a reusable idea for any streaming-generation setting where the conditioning signal itself repeats (e.g. cyclic motions, repeated camera angles, loop-heavy content)."
    limitations: "The paper states LiveAnimate currently generates at **480x480** resolution with three denoising steps, which limits visual fidelity and resolution. It also states the system does not support multi-person scenes or large camera motion."
    future_work: "The paper states extending LiveAnimate toward higher-resolution streaming generation, multi-person animation, and camera-dynamic scenes is left to future work."
    resources: "No public release verified. The paper does not mention a code, model, or project-page release in the sections read for this review."
thread:  # 3-4 blank-line-separated paragraphs: lineage → conceptual shift → what it opens
  ko: |-
    첫 문단 — lineage: 스트리밍·실시간 비디오 디퓨전에는 이 사이트가 다뤄온 자기 계보가 있다 — Self-Forcing/CausVid 계열은 양방향 비디오 디퓨전 트랜스포머를 블록-코잘 자기회귀 생성기로 바꾸고, 몇 단계의 샘플링으로 증류한 뒤, 스트림이 길어져도 생성 비용이 일정하도록 롤링 KV 캐시 윈도우를 유지한다. 이는 이미 제품 규모의 응용까지 도달했으며, 모두 동일한 최신성 기반 롤링 윈도우 캐시를 공유한다.

    둘째 문단 — shift: LiveAnimate의 전환은 좁지만 실질적이다 — 그 계보의 블록-코잘+증류 레시피는 그대로 두고, 캐시 축출 정책 자체를 다시 짠다. Pose-Retrieval Sink Attention은 '유지한다 = 최근에 유지됐다'는 가정을 버린다 — 지금 생성 중인 포즈가 이전에 본 포즈와 일치한다는 이유로 특정 과거 블록을 검색해 오는 두 번째 콘텐츠 주소 지정 싱크를 추가하는데, 이는 이 계보의 모든 최신성·중요도 기반 축출 규칙과는 다른 축(다시 필요해질 것은 무엇인가)이다.

    셋째 문단 — what it opens: 이 해법이 모델이나 증류 레시피가 아니라 캐시 관리 계층에만 존재하기 때문에, 포즈 기반 애니메이션뿐 아니라 조건 신호가 반복되는 모든 스트리밍 생성 설정 — 주기적이거나 반복적인 입력이 있어 '이 입력이 지난번 나타났을 때 무슨 일이 있었는가'가 '가장 최근에 무슨 일이 있었는가'보다 더 유용한 정보가 되는 모든 비디오·오디오·에이전트 제어 스트림 — 으로 전이될 만한 템플릿이 된다.
  en: |-
    Streaming, real-time video diffusion has its own lineage on this site — the Self-Forcing/CausVid line converts a bidirectional video diffusion transformer into a block-causal autoregressive generator, distills it to a handful of sampling steps, and keeps a rolling KV-cache window so generation cost stays bounded as the stream grows. This has already reached product-scale applications, all sharing the same recency-based rolling-window cache.

    LiveAnimate's shift is narrow but real: it keeps the block-causal-plus-distillation recipe from that lineage unchanged, and instead rebuilds the cache-eviction policy itself. Pose-Retrieval Sink Attention drops the assumption that keep should only ever mean kept recently — it adds a second, content-addressed sink that retrieves a specific past block because the pose being generated now matches a pose seen before, a different axis (what will be needed again) from every recency- or saliency-based eviction rule in this lineage.

    Because the fix lives entirely in the cache-management layer rather than the model or the distillation recipe, it is a template that should transfer to any streaming-generation setting where the conditioning signal recurs — not just pose-driven animation, but any video, audio, or agent-control stream where a periodic or repeating input makes what-happened-last-time-this-recurred a more useful thing to remember than what-happened-most-recently.
sparks:
  - ko: "논문은 3단계 디노이징에서 480×480 출력을 현재의 한계로 보고하며, 더 높은 해상도의 스트리밍 생성은 향후 과제로 남겨두었다 — 블록 해상도가 올라가면 PR-Sink의 유한 캐시 설계(블록 수, 싱크 크기)는 얼마나 바뀌어야 하고, 비용 증가는 블록당 디노이징 단계 자체에만 온전히 떨어질까?"
    en: "The paper reports 480x480 output at three denoising steps as a current ceiling, with higher-resolution streaming generation left to future work — how much of PR-Sink's bounded-cache design (block counts, sink sizes) would need to change if the block resolution went up, versus how much of the cost increase would fall entirely on the per-block denoising step itself?"
  - ko: "저자들은 다인물 장면이 현재 지원되지 않는다고 밝히고 이를 향후 과제로 명시한다 — 공유된 하나의 PR-Sink가 추적 대상 인물마다 하나씩의 싱크로 분화되어야 할까, 아니면 포즈 검색을 프레임 단위가 아니라 인물 단위로 키를 잡는다면 단일 Dynamic Sink로도 여러 피사체에 걸친 과거 맥락을 검색할 수 있을까?"
    en: "The authors flag that multi-person scenes are not currently supported and name it as future work — would a single shared PR-Sink need to become one sink per tracked person, or could a single Dynamic Sink retrieve historical context across multiple subjects if pose retrieval were keyed per-person rather than per-frame?"
source: "autosweep"
---

## Notes

<!-- optional free-form notes; the structured 13-item analysis lives in the frontmatter. -->
