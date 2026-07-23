---
title: "Self Gradient Forcing: Native Long Video Extrapolation"
arxivId: "2607.20368"
authors: "Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, Yaowei Li, Yawen Luo, Yijun Liu, Weiyang Jin, Songchun Zhang, Xianglong He, Xuying Zhang, Haoran Li, Haoyang Huang, Zeyue Xue, Nan Duan"
lab: "Joy Future Academy, JD"
date: 2026-07-23
tags: ["kv-cache", "video-diffusion", "training"]
topic: 'kv-cache'
summary: "Self Gradient Forcing names the historical context-gradient gap in Self-Forcing-style autoregressive video diffusion training — the historical KV cache is written once and then frozen, so future losses can never supervise how it should have been written — and fixes it with a two-pass training scheme that lets models trained on 5-second clips extrapolate natively to 240-second video."
summary_ko: "Self Gradient Forcing는 Self-Forcing 방식의 자기회귀 비디오 확산 학습에서 과거 KV 캐시가 한 번 쓰이고 나면 그대로 얼어붙어, 이후 손실이 그 쓰기 연산을 감독할 수 없다는 historical context-gradient gap을 명명하고, 5초짜리 클립으로 학습한 모델이 240초 비디오까지 자연스럽게 확장되게 하는 2단계 학습 기법으로 이를 해결한다."
links: ["surprise-forcing", "headcast"]
resources:
  - label: 'arXiv'
    url: 'https://arxiv.org/abs/2607.20368'
  - label: 'Project page'
    url: 'https://zhuang2002.github.io/SelfGradientForcing'
figures:
  - src: /figures/self-gradient-forcing/fig1.png
    caption: "Long-horizon consistency comparison under the same prompt, seed, and initialization: ordinary Self Forcing exhibits view jumps, scene breaks, and identity drift over 240 seconds, while Self Gradient Forcing preserves subject identity, pose, and scene layout."
    caption_ko: "동일한 프롬프트·시드·초기화 조건에서의 장기 일관성 비교 — 기존 Self Forcing은 240초에 걸쳐 시점 점프·장면 붕괴·정체성 드리프트를 보이는 반면, Self Gradient Forcing은 피사체 정체성·포즈·장면 배치를 유지한다."
    credit: "Figure 1 from arXiv:2607.20368 — authors' figure"
  - src: /figures/self-gradient-forcing/fig2.png
    caption: "From frozen-cache Self Forcing to Self Gradient Forcing: Self Forcing treats historical K/V entries as detached cache state, while SGF adds a parallel reconstruction pass that re-encodes the self-generated context with gradients flowing back into the K/V-writing computation."
    caption_ko: "고정 캐시 Self Forcing에서 Self Gradient Forcing으로 — Self Forcing은 과거 K/V 항목을 분리된(detached) 캐시 상태로 다루지만, SGF는 자기 생성 컨텍스트를 다시 인코딩하며 그래디언트가 K/V 쓰기 연산까지 흘러가게 하는 병렬 재구성 패스를 추가한다."
    credit: "Figure 2 from arXiv:2607.20368 — authors' figure"
analysis:
  ko:
    background: "자기회귀(AR) 비디오 확산 모델은 긴 영상을 블록/청크 단위로 순차 생성하며, 이전에 생성한 프레임들을 KV 캐시로 저장해 재사용한다. **Self Forcing**은 학습 시 실제 영상 대신 모델 자신이 생성한 롤아웃 히스토리를 조건으로 사용해, 학습-추론 간 노출 편향(exposure bias)을 줄이는 방식으로 최근 AR 비디오 확산 학습의 표준 레시피가 되었다."
    problem: "Self Forcing은 모델이 자기 생성 히스토리를 **읽는(read)** 방식은 학습시키지만, 그 히스토리가 처음에 K/V로 **쓰이는(write)** 연산은 감독하지 못한다 — 캐시는 정결(clean) 컨텍스트 타임스텝에서 한 번 계산되어 이후 청크에는 고정된(frozen) 상태로만 취급되기 때문이다. 저자들은 이를 **historical context-gradient gap**이라 부른다: 같은 causal DiT가 여러 타임스텝의 파라미터를 공유하기 때문에, 노이즈가 낀 디노이징 스텝에서의 업데이트가 감독되지 않는 정결 캐시-쓰기 연산을 조용히 드리프트시킬 수 있다."
    prior_limits: "직접적인 해법은 과거 KV 캐시를 미분 가능하게 유지해 미래 손실이 정결 타임스텝의 쓰기 연산까지 역전파되게 하는 것이지만, 이는 모든 과거 캐시 쓰기에 대한 autograd 그래프를 미래 청크가 소비할 때까지 살려둬야 한다는 뜻이다 — 롤아웃 길이·트랜스포머 깊이·순차적 캐시 업데이트에 따라 그래프가 계속 커져 확장이 어렵다."
    goal: "전체 순차 롤아웃을 통한 완전한 역전파 없이, 자기 생성 히스토리를 미래에 더 유용한 K/V로 쓰는 연산에 미래 손실의 감독 신호를 되돌려주는 것이 목표다."
    method: "**Self Gradient Forcing(SGF)**은 2-패스 학습 기법이다. Pass 1은 그래디언트 없이 실제 순차 AR 롤아웃을 추론과 동일하게 수행하며, 샘플링된 하나의 디노이징 종료 스텝에서 자기 생성 컨텍스트와 그때 모델에 입력된 노이즈 낀 잠재변수를 기록한다. Pass 2는 롤아웃 캐시를 버리고, 기록된 컨텍스트를 stop-gradient 정결 잠재변수 입력으로 삼아 그 종료 스텝의 컨텍스트 K/V 표현과 미래→컨텍스트 causal attention을 **병렬로 재계산**한다 — 마치 긴 촬영 영상 전체를 편집용으로 계속 붙들고 있는 대신, 문제가 된 한 장면만 골라 세트를 다시 세우고 재촬영해 그 장면에 대한 피드백을 반영하는 것과 비슷하다."
    key_idea: "핵심은 \"모델이 자기 생성 히스토리를 보는가\"(Self Forcing이 이미 해결)와 \"미래 손실이 그 히스토리를 K/V로 쓴 연산에 도달하는가\"(SGF가 복원)를 분리한 것이다. 새로운 손실 항을 추가하는 정규화가 아니라, 계산 그래프 상에서 그래디언트가 흐르는 경로 자체를 바꾸는 **credit-assignment 수정**이며, 길이에 무관하게 비용이 고정된 병렬 재구성으로 이를 구현한다."
    validation: "VBench 계열 지표(미학·배경·역동성·화질·모션 매끄러움·피사체 일관성·플리커링)로 60초·240초 두 지평선, 세 가지 초기화(Causal ODE/Causal CD/TF), frame-wise·chunk-wise 두 생성 방식을 모두 평가한다. 정성적 비교와 두 패스 재구성의 충실도, VAE 경계 효과에 대한 부가 분석도 포함한다."
    results: "TF 초기화 240초 설정에서 피사체 일관성 0.965→0.972, 배경 일관성 0.965→0.968처럼 대부분의 품질·일관성 지표에서 일관된 개선을 보인다. Dynamic Degree만 예외적으로 기존 Self Forcing이 더 높게 나오는 경우가 있는데, 저자들은 이를 장기 롤아웃에서 발생하는 장면 점프·카메라 기하 붕괴가 만드는 크지만 비일관적인 겉보기 움직임 때문이라고 해석한다(독립적인 모션 전용 지표로 검증되지는 않음). 5초 학습 윈도우로 학습한 모델이 240초(분 단위) 영상까지 자연스럽게 확장되는 것이 핵심 결과다."
    comparison: "동일한 프롬프트·시드·초기화·추론 구성 아래 고정 캐시(frozen-cache) Self Forcing과 짝지어 비교하며, SGF는 경쟁 기법이 아니라 기존 Self-Forcing 레시피 위에 그대로 얹을 수 있는 **직교적(orthogonal)** 보강으로 제시된다."
    significance: "효율적 비디오 생성 관점에서, 이 논문은 자기-롤아웃 학습에서 장기 드리프트를 단순한 용량이나 데이터 문제가 아니라 **크레딧 할당(credit assignment)** 문제로 재구성한다 — 캐시를 누가 쓰든 그 쓰기 연산이 미래 손실의 감독을 받는지가 핵심이라는 것이다. 이 프레임은 비디오뿐 아니라 자기 생성 상태를 캐싱하는 다른 AR 생성 설정에도 원칙적으로 적용 가능하다."
    limitations: "Dynamic Degree 이상 현상은 정성적으로만 해석되며, 독립적인 모션 전용 지표로 확정되지는 않는다(리뷰어 판단). 2-패스 재구성이 이론상 불가능한 완전한 순차 그래디언트를 얼마나 충실히 근사하는지는 이 리뷰가 읽은 범위(초록·서론·실험 섹션)에서는 정량적으로 명시되지 않는다 — 논문 내 명시 없음(부록 D에서 다룰 가능성 있음, 이 리뷰는 미확인)."
    future_work: "이 리뷰가 읽은 범위에서 저자들이 명시한 future-work 항목은 확인되지 않는다 — 논문 내 명시 없음."
    resources: "코드와 모델은 프로젝트 페이지에 공개 예정이라고 명시되어 있으나, 이 리뷰 작성 시점에는 저장소 내용 자체가 아직 확인되지 않았다."
  en:
    background: "Autoregressive (AR) video diffusion models generate long video block-by-block, reusing previously generated frames via a KV cache. **Self Forcing** has become the standard recent training recipe: it trains the student on histories produced by its own rollout rather than ground-truth video, reducing the training-inference exposure-bias mismatch."
    problem: "Self Forcing trains the model to **read** self-generated history, but never supervises the computation that originally **writes** that history into K/V — the cache is computed once at the clean context timestep and treated as frozen state by every later chunk. The authors name this the **historical context-gradient gap**: because the same causal DiT shares parameters across timesteps, updates from noisy denoising-step losses can silently drift the unsupervised clean-timestep cache-writing computation."
    prior_limits: "A direct fix would keep the historical KV cache differentiable so future losses backpropagate into the clean-timestep writing computation — but that requires retaining the autograd graph for every historical cache write until future chunks consume it, a graph that grows with rollout length, transformer depth, and sequential cache updates, making it impractical to scale."
    goal: "Restore the missing gradient supervision for how self-generated history gets written into more useful K/V representations, without full backpropagation through the entire serial rollout."
    method: "**Self Gradient Forcing (SGF)** is a two-pass training strategy. Pass 1 performs the true no-gradient serial AR rollout exactly as at inference, recording the self-generated context and the noisy latents fed to the model at one sampled denoising exit step. Pass 2 discards the rollout cache and, using the recorded context as stop-gradient clean-latent input, **recomputes that same exit-step computation in parallel** — the context K/V representations and future-to-context causal attention — so losses on future latents can now backpropagate into the cache-writing weights. It is a bit like not keeping an entire long shoot's raw footage on hand for editing, but instead re-building and re-shooting just the one contested scene in parallel, so notes from later footage can actually change how that scene was staged."
    key_idea: "The core move is separating two questions that Self Forcing conflates: whether the model *sees* its own generated history (already fixed) versus whether future losses *reach* the computation that wrote that history into K/V (restored here). This is a **credit-assignment fix**, not a new regularization loss — it changes which computation the gradient graph touches, implemented as a bounded, rollout-length-independent parallel reconstruction."
    validation: "VBench-family metrics (aesthetics, background, dynamics, imaging, motion smoothness, subject consistency, flickering) are evaluated at 60-second and 240-second horizons, across three initializations (Causal ODE / Causal CD / TF), for both frame-wise and chunk-wise generation. Qualitative comparisons, two-pass reconstruction fidelity, and a VAE start-boundary analysis are also reported."
    results: "Most quality and consistency metrics improve consistently — e.g. under TF initialization at 240s, subject consistency rises 0.965→0.972 and background consistency 0.965→0.968. Dynamic Degree is the one exception where plain Self Forcing sometimes scores higher; the authors attribute this to long rollouts accumulating scene jumps and broken camera geometry that inflate apparent motion without being genuinely better (not confirmed against an independent motion-only metric). The headline generalization result: models trained on only 5-second clips extrapolate natively to 240-second (minute-scale) video."
    comparison: "SGF is compared pairwise against matched frozen-cache Self Forcing under identical prompts, seeds, initialization, and inference geometry, and is explicitly positioned as **orthogonal** — a drop-in addition to any existing TF/CD/ODE-initialized Self-Forcing checkpoint, not a competing training recipe."
    significance: "From an efficient-video-generation standpoint, this paper reframes long-horizon drift in self-rollout training as a **credit-assignment** problem rather than a capacity or data problem — what matters is whether the cache-writing computation, whoever performs it, is supervised by future losses. That framing applies in principle beyond video, to any autoregressive generation setup that caches its own self-generated state."
    limitations: "The Dynamic-Degree anomaly is interpreted qualitatively rather than resolved with an independent motion-only metric (reviewer judgment). How tightly the two-pass parallel reconstruction approximates the true, infeasible full serial gradient is not quantified within the scope this review draws from (abstract, introduction, and experiments sections) — not stated in the paper at that depth (Appendix D may address this; not verified here)."
    future_work: "No explicit future-work items are identifiable within the scope this review draws from — not stated in the paper."
    resources: "Code and models are stated to be released on the project page, but the repository contents themselves were not yet verifiable at the time of this review."
thread:
  ko: |-
    자기회귀 비디오 확산에서 KV 캐시는 지금까지 대체로 "무엇을 저장/제거할 것인가"라는 추론-시점 질문으로 다뤄져 왔다 — Surprise Forcing의 서프라이즈 기반 재수용, HeadCast의 헤드별 라우팅([[headcast]] 참고)이 모두 이 계열이다. 이 논문이 서 있는 자리는 조금 다르다: Self Forcing이라는, 자기 생성 히스토리로 학습해 노출 편향을 줄이는 학습 레시피 계열 위에서, "캐시에 무엇이 남는가"가 아니라 "캐시가 애초에 어떻게 쓰이도록 학습되는가"를 묻는다.

    전환은 크레딧 할당(credit assignment) 축으로의 이동이다. Self Forcing은 모델이 자기 생성 히스토리를 읽는 법은 학습시키지만, 그 히스토리를 K/V로 쓰는 연산에는 미래 손실이 닿지 않는다는 비대칭을 남겨둔다. 이 논문은 그 비대칭에 이름을 붙이고(historical context-gradient gap), 완전한 역전파는 감당할 수 없다는 제약 아래 "얼마나 감독할 것인가"와 "얼마나 계산 비용을 쓸 것인가" 사이의 절충을 2-패스 병렬 재구성으로 풀어낸다 — 순차적으로 그래프를 통째로 유지하는 대신, 샘플링된 한 스텝만 병렬로 다시 계산해 그래디언트 경로를 되살리는 방식이다.

    이 논문이 여는 질문은 크레딧 할당 프레임이 얼마나 일반적인가이다: 자기 생성 상태를 캐싱하는 다른 자기회귀 생성 설정(텍스트 dLLM의 커밋된 KV, 오디오 스트리밍 등)에도 같은 "쓰기 연산이 감독받지 않는다"는 구조가 숨어 있을까? 그리고 추론-시점 캐시 관리(Surprise Forcing, HeadCast류)와 학습-시점 크레딧 할당(SGF)은 서로 배타적이지 않다 — 같은 체크포인트 위에 두 축을 함께 쌓을 수 있는지가 다음으로 자연스럽게 따라오는 질문이다.
  en: |-
    KV-cache handling in autoregressive video diffusion has mostly been framed as an inference-time question of what to keep or discard — Surprise Forcing's surprise-gated re-admission and HeadCast's per-head routing (see [[headcast]]) both sit in that lineage. This paper stands somewhere different: on top of the Self-Forcing family of training recipes, which train on self-generated rollout history to reduce exposure bias, it asks not "what stays in the cache" but "how does the cache get trained to be written in the first place."

    The shift is a move onto the credit-assignment axis. Self Forcing trains the model to read its own generated history but leaves an asymmetry where future losses never reach the computation that writes that history into K/V. This paper names that asymmetry (the historical context-gradient gap) and, under the constraint that full backpropagation through the rollout is infeasible, resolves the trade-off between "how much to supervise" and "how much compute to spend" with a two-pass parallel reconstruction — instead of keeping the whole serial graph alive, it recomputes just one sampled step in parallel to restore the gradient path.

    The question this leaves open is how general the credit-assignment frame is: does the same "the write computation goes unsupervised" structure hide in other autoregressive generation setups that cache self-generated state — committed KV in text dLLMs, streaming audio, and so on? And inference-time cache management (Surprise-Forcing/HeadCast-style) and training-time credit assignment (SGF) are not mutually exclusive — whether both axes can be stacked on the same checkpoint is a natural next question.
sparks:
  - ko: "논문은 Pass 2의 병렬 재구성이 참된(그러나 계산 불가능한) 순차 그래디언트를 얼마나 충실히 근사하는지 별도로 정량화하지 않는다 — 근사 오차를 직접 측정하는 진단(예: 짧은 롤아웃에서만 가능한 완전 역전파와의 직접 비교)을 설계하면 이 방법의 적용 한계를 더 명확히 할 수 있을 것이다."
    en: "The paper doesn't separately quantify how tightly Pass 2's parallel reconstruction approximates the true (but intractable) serial gradient — a diagnostic that directly measures this approximation error (e.g. comparing against full backprop on short-enough rollouts where it's feasible) would sharpen the boundaries of when this method's fix actually helps."
  - ko: "Dynamic Degree 이상 현상은 저자들이 정성적으로만 설명하고 넘어간다 — 장면 붕괴로 인한 겉보기 움직임과 진짜 모션 품질을 분리하는 독립적인 지표를 도입하면, 이 논문이 스스로 제기한 해석을 검증할 수 있다."
    en: "The authors explain the Dynamic-Degree anomaly only qualitatively — introducing an independent metric that separates apparent motion from scene breaks from genuine motion quality would let this paper's own interpretation actually be tested."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
