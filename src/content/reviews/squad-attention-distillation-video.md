---
title: "SQuad: Sub-Quadratic Attention Distillation for Efficient Video Generation"
arxivId: "2608.16585"
date: "2026-08-24"
tags:
  - video-diffusion
  - attention-distillation
  - few-step-distillation
topic: compression
summary: SQuad distills a pretrained video diffusion transformer's full-softmax attention into a fixed, sub-quadratic O(n*sqrt(n)) factorization -- a local pass within sqrt(n)-sized windows plus a global cross-window pass, still true softmax throughout -- and pairs it with step distillation, matching the original's quality at 6 sampling steps instead of 100 while cutting attention compute by roughly 67x.
summary_ko: 비디오 확산 트랜스포머의 풀-소프트맥스 어텐션을, sqrt(n) 크기 윈도 내 지역 연산과 윈도 간 전역 연산 두 단계로 이뤄진 고정된 서브-쿼드러틱 O(n*sqrt(n)) 구조로 증류하는 논문. 소프트맥스는 그대로 유지하면서, 100 스텝이던 샘플링을 6 스텝으로 줄이고도 원본 품질을 유지하며 어텐션 연산량은 약 67배 줄인다.
links:
  - sparsepr-video-sparse-attention
  - context-matched-distillation
  - pdd-parallel-decoding-distillation
  - self-gradient-forcing
resources:
  - label: arXiv abstract
    url: https://arxiv.org/abs/2608.16585
  - label: arXiv PDF
    url: https://arxiv.org/pdf/2608.16585
figures: []
analysis:
  ko:
    background: 비디오 확산 트랜스포머(DiT)는 프레임과 해상도가 늘어날수록 잠재 토큰 수 n이 커지고, 셀프 어텐션 연산량이 그 제곱(O(n^2))으로 늘어난다. 오늘날 하드웨어에서는 이 어텐션 비용이 전체 생성 예산을 지배해, 한 번에 생성할 수 있는 해상도와 길이를 사실상 결정짓는다.
    problem: 어텐션을 값싸게 만드는 두 갈래 접근 모두 흠이 있다. 소프트맥스를 선형/저랭크 근사로 완전히 대체하면 연산은 줄지만 원본의 표현력을 좀처럼 회복하지 못해 품질 격차가 남고, 반대로 정확한 소프트맥스를 유지한 채 일부 토큰쌍만 계산하는 스파스 마스크 방식은 품질은 지키지만 학습 가능한 라우팅 로직이나 추가 파라미터(72-283M) 같은 복잡성을 끌고 온다.
    prior_limits: 선형/하이브리드 대체 계열(Attention Surgery, ReHyAt)은 결국 소프트맥스 자체를 포기하는 방향이라 표현력 손실이 구조적이다. 스파스-마스크 계열(VSA, Radial Attention, Jenga)은 소프트맥스는 지키지만 『어떤 토큰쌍을 남길지』를 데이터에 따라 학습하거나 고정된 감쇠 규칙으로 정해야 해서, 방식이 복잡해지거나 하드웨어 친화적이지 않은 커널이 필요해진다.
    goal: "목표는 두 가지를 동시에 만족하는 것이다: (1) 진짜 소프트맥스를 전 구간에서 유지하면서 (2) 학습된 라우팅이나 추가 파라미터 없이 고정된 구조만으로 O(n*sqrt(n)) 복잡도를 달성하는 것. 여기에 더해, 어텐션 자체의 비용뿐 아니라 샘플링에 필요한 스텝 수(NFE)까지 함께 줄이는 것도 목표에 포함된다."
    method: "SQuad-Attention은 두 단계로 구성된다: 먼저 각 토큰이 O(sqrt(n)) 크기의 지역 윈도 안에서만 서로 주의를 기울이고(지역 패스), 그다음 각 윈도에서 같은 위치에 있는 토큰들끼리 모든 윈도를 가로질러 다시 주의를 기울인다(전역 패스) -- 두 패스 모두 진짜 소프트맥스를 쓴다. 이 구조를 사전학습된 Wan 2.2 5B DiT에 이식하기 위해, Flow-Matching SFT로 먼저 적응시킨 뒤 DMD2(분포 매칭 증류)로 다시 다듬는다. DMD2는 부수적으로 샘플링 스텝(NFE)도 100에서 6으로 줄여준다."
    key_idea: 핵심 통찰은 『지역 윈도 + 윈도 간 전역 교환』이라는 고정되고 구조화된 통신 패턴 하나로 O(n^2)을 O(n*sqrt(n))까지 낮추면서도 소프트맥스를 포기하지 않아도 된다는 것이다. 마치 큰 회의를 없애는 대신, 작은 팀별 회의(지역 패스)로 논의를 정리한 뒤 팀 대표들끼리 다시 모여(전역 패스) 결론을 맞추는 것과 비슷하다 -- 전체를 한 번에 모으지 않고도 전역 정보 흐름은 유지된다.
    validation: Wan 2.2 5B(주력 모델, 전체 증류 수행)와 Wan 2.1 1.3B(전체 증류)에서 검증했고, Wan 2.1 14B에서는 증류 없이 효율성 수치만 확인해 규모가 커져도 같은 경향이 유지되는지 봤다. 품질은 VBench 지표와 사람 선호도 조사로, 속도는 CUDA 이벤트 훅을 이용한 실제 순전파 지연시간(eager 모드와 torch.compile 모드 둘 다)으로 측정해, FLOP 수치만으로 그친 주장이 아니다.
    results: Wan 2.2 5B에서 SQuad는 원본 대비 VBench 총점이 사실상 동일(**83.20 대 83.08**)하면서, 블록당 어텐션 FLOPs를 약 **67배**, 어텐션 지연시간을 약 **11배** 줄였고, 블록 지연시간은 62.01ms→19.04ms(**3.3배**), 종단간 DiT 지연시간은 eager 모드 870ms→520ms·컴파일 모드 667ms→314ms(**2배**)로 줄었다. 샘플링 스텝은 **100에서 6**으로 줄었다. Wan 2.1 14B에서도 41.959→20.223 TFLOPs, 300.29ms→59.24ms로 같은 확장 경향(토큰 그리드가 클수록 이득이 커짐)이 그대로 확인됐다.
    comparison: "VSA, Jenga, Radial Attention, Attention Surgery, ReHyAt 등 네 기준선과 비교했을 때, SQuad는 추가 파라미터 없이(비교 대상 중 세 개는 72-283M 파라미터를 추가) 특수 GPU 커널도 없이 지연시간과 TFLOPs에서 앞선다. 다만 정직하게 밝혀진 예외가 있다: Radial Attention은 VBench 총점(84.56)에서 SQuad(83.20)보다 오히려 높다 -- 즉 SQuad의 우위는 『무파라미터·효율성』 축에서의 파레토 우위이지, 무조건적인 품질 1위는 아니다. 사람 선호도 조사에서도 SQuad 선호 35% 대 DMD+Radial Attention 선호 33%, 무응답 31%로 사실상 백중세였다."
    significance: 효율적 AI 관점에서 이 논문의 의미는, 『소프트맥스를 버리는 근사』와 『소프트맥스는 지키되 복잡한 학습된 마스킹을 추가하는 근사』 사이에 세 번째 길 -- 소프트맥스를 유지한 채 고정된 구조만으로 서브-쿼드러틱을 달성하는 길 -- 이 실제로 작동함을 보여준 데 있다. 또한 어텐션 복잡도 축소(SQuad)와 스텝 수 축소(DMD2)라는 서로 다른 두 효율화 축이 경쟁하지 않고 하나의 증류 파이프라인 안에서 함께 결합될 수 있음을 보여준다.
    limitations: "논문이 스스로 밝힌 한계: 어텐션 구조 변경이 DMD2 스텝 증류와 항상 함께 적용돼, 스텝 수를 줄이지 않는 순수 Flow-Matching 환경에서 어텐션 변경만의 효과를 분리한 실험이 없다. 지역→전역 두 패스만 탐색했고, 두 개보다 많은 패스를 쌓는 조합은 시도하지 않았다. 비디오 도메인에만 적용했고 다른 모달리티로의 전이는 다루지 않는다. (리뷰어 판단) 이 논문의 대상은 전체 잠재 공간에 양방향으로 접근하는 비-인과적(non-causal) 비디오 DiT로, KV 캐시나 인과적(causal) 자기회귀 비디오 생성 세팅은 전혀 다루지 않는다."
    future_work: "논문이 직접 밝힌 다음 단계: (1) DMD2와 결합을 풀어 순수 Flow-Matching 상태에서 어텐션 변경만의 효과를 분리 측정하는 것, (2) 지역/전역 두 패스보다 많은 패스를 쌓아 깊이와 비용/표현력을 다시 트레이드오프하는 것, (3) 비디오를 넘어 다른 도메인·모달리티로 이 아이디어를 확장하는 것."
    resources: 논문 본문과 PDF 전체를 확인했으나 프로젝트 페이지나 코드 저장소 링크는 발견하지 못했다. 공개 코드 저장소는 확인 안 됨.
  en:
    background: Video diffusion transformers (DiTs) pay a self-attention cost that grows quadratically (O(n^2)) with the number of latent tokens n, and n grows fast with resolution and frame count. On current hardware this attention cost dominates the entire generation budget, effectively capping the resolution and duration a model can produce in one pass.
    problem: Both standard ways of making attention cheaper have a flaw. Replacing softmax outright with a linear/low-rank surrogate cuts cost but rarely recovers the original expressivity, leaving a quality gap. Keeping the exact softmax but computing only a subset of token pairs (sparse masking) preserves quality but drags in complexity -- learned, data-dependent routing or extra parameters (72-283M in some baselines).
    prior_limits: The linear/hybrid-surrogate line (Attention Surgery, ReHyAt) gives up softmax itself, so the expressivity loss is structural. The sparse-mask line (VSA, Radial Attention, Jenga) keeps softmax but must decide which token pairs to keep via learned routing or fixed decay rules, which either adds complexity or requires specialized, less hardware-friendly kernels.
    goal: "The goal is to satisfy two things at once: (1) keep a genuine softmax throughout, and (2) reach O(n*sqrt(n)) complexity using only a fixed structure -- no learned routing, no added parameters. On top of that, the goal also covers cutting the number of sampling steps (NFE) needed, not just the per-step attention cost."
    method: "SQuad-Attention has two passes: first, each token attends only within an O(sqrt(n))-sized local window (local pass); then, tokens at the same position within each window attend to each other across all windows (global pass) -- both passes use a genuine softmax. To fit this structure onto a pretrained Wan 2.2 5B DiT, the model is first adapted with Flow-Matching SFT, then refined with DMD2 (distribution-matching distillation), which as a side effect also collapses the sampling step count (NFE) from 100 to 6."
    key_idea: "The core insight is that a single fixed, structured communication pattern -- local windows plus a cross-window global exchange -- can bring O(n^2) down to O(n*sqrt(n)) without giving up softmax at all. It is a bit like replacing one giant plenary meeting with small team huddles (the local pass) followed by a meeting of team representatives (the global pass): global information still flows, just without ever convening everyone at once."
    validation: Validated on Wan 2.2 5B (the main model, fully distilled) and Wan 2.1 1.3B (fully distilled), with Wan 2.1 14B checked for efficiency numbers only (no distillation run at that scale) to confirm the trend holds as scale grows. Quality is measured via VBench and a human preference study; speed is measured as real forward-pass latency (CUDA-event hooks, both eager and torch.compile modes) rather than FLOP counts alone.
    results: "On Wan 2.2 5B, SQuad matches the original VBench total almost exactly (**83.20 vs. 83.08**) while cutting per-block attention FLOPs by roughly **67x** and attention latency by roughly **11x**; block latency drops from 62.01ms to 19.04ms (**3.3x**), and end-to-end DiT latency drops from 870ms to 520ms in eager mode and 667ms to 314ms compiled (**2x**). Sampling collapses from **100 steps to 6**. The same scaling signature (bigger payoff at larger token grids) is confirmed on Wan 2.1 14B: 41.959 -> 20.223 TFLOPs, 300.29ms -> 59.24ms."
    comparison: "Against four baselines (VSA, Jenga, Radial Attention, Attention Surgery, ReHyAt), SQuad wins on latency and TFLOPs with zero added parameters (three of the baselines add 72-283M) and no specialized GPU kernels. One honest exception the paper itself reports: Radial Attention actually posts a higher VBench total (84.56) than SQuad's 83.20 -- so SQuad's advantage is a Pareto win on the no-added-parameters/efficiency axis, not an outright best-quality claim. A human preference study came out close to even too: 35% preferred SQuad, 33% preferred the DMD+Radial-Attention baseline, 31% had no preference."
    significance: From an efficient-AI lens, this paper's contribution is showing that a third path exists between giving up softmax entirely and keeping softmax but adding complex learned masking -- a fixed, structured factorization that keeps genuine softmax and still reaches sub-quadratic cost. It also shows that two distinct efficiency axes -- attention complexity (SQuad) and step count (DMD2) -- can be collapsed into one distillation pipeline rather than competing for the same training budget.
    limitations: "Paper-stated: the attention change is coupled to DMD2 step-distillation throughout, so no experiment isolates the attention change alone under pure Flow-Matching without step reduction. Only two passes (local, global) were explored; stacking more than two was not tried. The method is evaluated only on video; transfer to other modalities is untested. (Reviewer judgment) This paper's scope is a non-causal video DiT with bidirectional access to the whole latent -- it says nothing about a KV-cache or causal/autoregressive video-generation setting."
    future_work: "The paper names three concrete next steps: (1) decoupling from DMD2 to isolate the attention change's effect under pure Flow-Matching, (2) composing more than two passes to trade a bit more depth for a cheaper or more expressive operator, and (3) extending the idea beyond video to other domains and modalities."
    resources: The full paper text and PDF were checked; no project page or code repository link was found. No public code release verified.
thread:
  ko: |-
    비디오 DiT의 어텐션을 값싸게 만드는 계열은 크게 둘로 갈려 왔다. 하나는 소프트맥스를 선형/하이브리드 커널로 바꿔치기하는 노선(Attention Surgery, ReHyAt)이고, 다른 하나는 소프트맥스는 지키되 일부 토큰쌍만 골라 계산하는 노선(VSA, Radial Attention, Jenga)이다. 이와는 별개로, CausVid·Self Forcing·DMD2로 이어지는 스텝-증류 계열은 어텐션 구조는 그대로 둔 채 샘플링 스텝 수(NFE)만 줄여왔다.

    SQuad는 이 두 노선 중 어느 쪽도 택하지 않는다. 소프트맥스를 버리지도, 학습된 라우팅을 추가하지도 않고, 『지역 윈도 + 윈도 간 전역 교환』이라는 고정된 구조만으로 서브-쿼드러틱 복잡도에 도달한다. 그리고 이걸 별도의 축이 아니라 스텝-증류(DMD2)와 같은 파이프라인 안에 묶어, 어텐션 비용 축소와 스텝 수 축소를 동시에 얻는다.

    이 결과가 일반화된다면, 다음 질문은 『학습된 라우팅 없이도 되는 고정 구조가 어디까지 통하는가』다. 논문 스스로도 밝히듯 두 패스보다 많은 패스를 쌓는 조합, 그리고 비디오를 넘어선 다른 모달리티로의 확장이 남은 열린 방향이다. 텍스트-어텐션이나 VAE 디코더 같은 다른 값비싼 연산에도 같은 『지역+전역 고정 패스』 아이디어가 통할지도 자연스러운 다음 질문이다.
  en: |-
    Making video-DiT attention cheaper has largely split into two lines: one replaces softmax with a linear/hybrid kernel (Attention Surgery, ReHyAt), the other keeps softmax but computes only a subset of token pairs (VSA, Radial Attention, Jenga). Separately, the step-distillation line (CausVid, Self Forcing, DMD2) leaves attention structure untouched and instead cuts the number of sampling steps (NFE).

    SQuad takes neither of the two attention-side routes. It does not give up softmax, and it does not add learned routing -- it reaches sub-quadratic complexity with a fixed local-window-plus-cross-window-global-exchange structure alone. And rather than treating this as a separate axis from step count, it folds both into one distillation pipeline (SFT then DMD2), buying attention-cost reduction and step-count reduction together.

    If this generalizes, the next question is how far a fixed, routing-free structure like this can go. The paper itself names composing more than two passes, and extending beyond video to other modalities, as open directions. A natural further question is whether the same fixed local-plus-global-pass idea would work on other expensive operations in the same pipeline -- text cross-attention, or the VAE decoder.
sparks:
  - ko: 논문은 DMD2 스텝 증류와 어텐션 구조 변경을 항상 함께 적용해, 순수 Flow-Matching(스텝 수 그대로)에서 어텐션 변경만의 효과를 분리한 실험이 없다고 밝힌다 -- 이 분리 실험을 하면 두 효율화 축(어텐션 복잡도, 스텝 수)이 정말 독립적으로 기여하는지 확인할 수 있을 것이다.
    en: The paper states that the attention change is always applied together with DMD2 step-distillation, with no experiment isolating the attention change alone under unchanged step count -- running that isolation would show whether the two efficiency axes (attention complexity, step count) really contribute independently.
  - ko: 논문은 지역-윈도/전역-교환 두 패스만 시도했고, 더 많은 패스를 쌓는 조합은 시도하지 않았다고 밝힌다 -- 세 패스 이상을 쌓으면 표현력과 비용의 트레이드오프가 어떻게 바뀌는지가 열린 질문이다.
    en: The paper says it only tried two passes (local, global) and did not try stacking more -- how the expressivity/cost trade-off shifts with three or more passes is an open question the paper names but does not answer.
source: autosweep
---

## Notes

<!-- Structured 13-item bilingual analysis lives in the frontmatter above. -->
