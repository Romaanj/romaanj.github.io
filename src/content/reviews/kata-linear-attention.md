---
title: "Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones"
arxivId: "2607.17419"
authors: "Ayoub Ghriss, Sourav Chakraborty"
date: 2026-07-21
tags: ["hybrid-architecture", "linear-attention", "associative-recall"]
topic: 'hybrid-architecture'
summary: "KATA derives linear-attention feature maps from symmetric-cone geometry, names a Welch-bound interference floor that governs associative-recall capacity, and shows several variants beating Gated DeltaNet on long-range recall while running up to ~11x FlashAttention-2's throughput at 131k tokens."
summary_ko: "KATA는 대칭 원뿔(symmetric cone) 기하학에서 선형 어텐션 피처맵을 유도하고, 결합 회상(associative recall) 용량을 지배하는 Welch-bound 간섭 하한을 명명하며, 여러 변형이 장거리 회상에서 Gated DeltaNet을 능가하면서 131k 토큰에서 FlashAttention-2 대비 최대 약 11배 처리량을 낸다는 것을 보인다."
links: ["gated-deltanet", "going-linear"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.17419' }
  - { label: 'GitHub', url: 'https://github.com/ayghri/kata' }
figures:
  - src: /figures/kata-linear-attention/fig1.png
    caption: "Forward throughput over batch-size x sequence-length: the associative-scan form of KATA, a sequential-scan baseline, and Gated DeltaNet compared on an NVIDIA H100."
    caption_ko: "배치크기×시퀀스길이에 따른 forward 처리량: KATA의 associative-scan 형태, 순차-스캔 베이스라인, Gated DeltaNet을 NVIDIA H100에서 비교."
    credit: "Figure 3 from arXiv:2607.17419 — authors' figure"
  - src: /figures/kata-linear-attention/fig2.png
    caption: "The PSD cone, one of three symmetric-cone geometries the paper uses to certify nonnegative attention weights — linearly isomorphic to the Lorentz cone, and the geometric root of KATA's rank-one PSD feature map."
    caption_ko: "논문이 어텐션 가중치의 비음수성을 보장하는 데 쓰는 세 가지 대칭 원뿔 기하 중 하나인 PSD 원뿔 — Lorentz 원뿔과 선형 동형이며, KATA의 rank-one PSD 피처맵의 기하학적 뿌리."
    credit: "Figure 1(c) from arXiv:2607.17419 — authors' figure"
analysis:
  ko:
    background: '선형 어텐션(linear attention)은 상수 시간 순환 추론(recurrent inference)을 약속하지만, 결합 회상(associative recall) — 이전에 본 키-값 쌍을 정확히 다시 불러오는 능력 — 에서 성능이 급격히 나빠지는 것으로 잘 알려져 있다. 이 논문은 이 오래된 약점을 "용량 벽(capacity wall)"이라 부르고 정면으로 다룬다.'
    problem: '기존 커널 피처맵들은 소프트맥스가 암묵적으로 갖는 비음수(nonnegative) 어텐션 가중치 구조를 대체로 임시방편적으로 근사해왔고, 왜 특정 피처맵이 회상 용량을 더 잘 보존하는지에 대한 원칙적인 설명이 없었다. 문제는 회상 실패를 그저 "선형 어텐션이 소프트맥스의 열등한 근사"라는 식으로 치부할 게 아니라, 정확히 어떤 기하학적 제약이 용량을 결정하는지 규명하는 것이다.'
    prior_limits: 'Gated DeltaNet 같은 델타-룰 순환은 반복되는 주소에서 오래된 내용을 지우는 게이팅으로 회상을 개선했지만, 그 개선이 어떤 용량 한계에 부딪히는지는 이론적으로 특정되지 않았다. 회상을 스펠컬 패킹(spherical packing) 문제로 정식화한 선행 연구도 실제 하드웨어 커널로 구현되고 실증적으로 소프트맥스 및 최신 선형 어텐션과 나란히 비교된 사례는 드물었다.'
    goal: '어텐션 가중치의 비음수성을 인증하는 기하학적 원리(대칭 원뿔)로부터 원칙적으로 피처맵을 유도하고, 그 기하가 허용하는 회상 용량의 한계를 정확히 수량화하며, 그 한계 바로 위에서 추가 파라미터 없이 상태를 키우는 실용적 방법을 찾는 것이 목표다.'
    method: '**KATA**(Kernelized Linear Attention Activations)는 자기쌍대 균질 원뿔(self-dual homogeneous cone)을 통해 비음수 어텐션 가중치를 인증하는 피처맵 계열을 유도한다. Rank-one PSD(양semidefinite) 피처가 용량-간섭 트레이드오프에서 유리함을 보이고, 파라미터 없는 컨벡스 출력 게이트를 복원하며, 회상 용량을 Welch bound 간섭 하한으로 특징짓는다. 이는 마치 제한된 진열 공간에 라벨이 흐려지지 않도록 최대한 많은 물건을 채우는 문제와 비슷하다 — 물건(키)을 너무 빽빽하게 넣으면 라벨(값)이 서로 뒤섞이기 시작한다.'
    key_idea: '핵심은 회상을 "스펠컬 패킹" 문제로 재정식화하면 소프트맥스의 지수적 날카로움(sharpening)과 선형 어텐션의 근사적 커널을 하나의 기하학적 틀 안에서 통일적으로 설명할 수 있다는 것이다. Welch bound 간섭 하한 위에서는 **파라미터를 추가하지 않고도** 상태를 키울 수 있는데, 사영 차원에서 지수적으로 많은 키를 담는 스펠컬 코드(spherical code)가 존재하기 때문이다.'
    validation: '장거리 MQAR(다중-질의 결합 회상)와 반복-키 덮어쓰기(repeated-key overwrite) 합성 태스크로 회상 능력을, NIAH·FDA·SQuAD로 문맥적 유창성을, 340M 파라미터 사전학습(word-level perplexity, zero-shot 정확도)으로 언어모델링 품질을 평가한다. 커널은 두 지점에서 벤치마크된다 — flash-attention 스타일 forward, 그리고 정확한 O(T) 청크-상태 형태.'
    results: '여러 KATA 변형이 장거리 MQAR·반복-키 덮어쓰기에서 **Gated DeltaNet을 능가**하며, **16배** out-of-distribution 길이 확장에서도 **0.985 MQAR**을 유지해 소프트맥스 대비 약 **1/4**의 KV 캐시만으로 근접한 성능을 낸다. 커널은 flash-attention 스타일 forward에서 FlashAttention-2 대비 **약 1.6배**, 청크-상태 형태는 131k 토큰에서 **약 11배** 처리량을 낸다. Associative scan은 청크 크기 C에 대해 청크 간 순환 깊이를 O(log(T/C))로 낮춰 순차 선형-어텐션 베이스라인 대비 평균 **2.4배** 처리량을 낸다.'
    comparison: 'KATA-M1은 더 큰 기하학적 패킹 예산(266,240개 엔트리 상태)을 갖지만, KATA-M2는 오히려 더 적은 135,168개 엔트리로 더 강한 hard-needle 회상을 얻는다 — 학습이 하나의 64차원 인자보다 두 개의 32차원 인자를 더 잘 배운다는 뜻이다. 즉 기하학적 용량 상한과 실제로 학습되는 용량은 별개이며, 이 논문은 둘을 명시적으로 분리해 보고한다.'
    significance: '효율적 아키텍처 설계 관점에서, 회상 실패를 "소프트맥스 대비 근사 오차"가 아니라 이름 붙은 물리량(간섭 하한)으로 다루면 어떤 피처맵이 왜 더 나은 용량을 갖는지 예측 가능해진다. 선형/하이브리드 어텐션이 소프트맥스와 진짜로 경쟁하려면 넘어야 할 회상 격차를 좁히는 이론적·실용적 도구를 동시에 제공한다.'
    limitations: '논문 자체가 "Discussion and Limitations" 절에서 명시한다 — KATA-M1은 NIAH와 direct-copy SWDE에는 강하지만, 문맥 해석과 상관된 후보 중 선택을 요구하는 FDA·SQuAD 같은 태스크에서는 성능이 떨어진다(KATA-M2와 DeltaKATA-M1이 그 격차의 일부를 회복). 기하학적 용량 상한은 실수·단위벡터·2차(degree-two) 모델이라는 특정 조건 아래서만 성립하는 "달성 가능한 상한"이라는 점도 저자들이 명시한다.'
    future_work: '논문이 네 가지 방향을 명시한다 — (1) 패킹 이론과 구성적 딕셔너리를 복소 공간으로 확장(구형 부호·ETF가 같은 차원에서 더 큰 용량을 줄 수 있음), (2) GatedKATA·GatedDeltaKATA의 체계적 사전학습과 DeltaKATA를 M1 이상으로 평가(모두 전용 융합 GPU 커널 필요), (3) KATA-M1의 전체 기하를 d_head=64 이상에서 학습 가능성 검증, (4) 텐서-트레인 분해로 전체 텐서 상태를 실체화하지 않고 더 높은 짝수차 외적 피처를 얻는 경로 모색.'
    resources: '공식 GitHub 저장소가 확인된다(ayghri/kata, curl로 200 응답 확인) — 코드가 공개되어 있다.'
  en:
    background: 'Linear attention promises constant-time recurrent inference, but it is well known to degrade sharply on associative recall — the ability to precisely retrieve a previously seen key-value pair. This paper names that long-standing weakness the "capacity wall" and attacks it head-on.'
    problem: 'Existing kernel feature maps have largely approximated softmax''s implicit nonnegative-attention-weight structure in an ad hoc way, without a principled account of why one feature map preserves recall capacity better than another. The problem is to stop treating recall failure as merely "linear attention is a worse approximation of softmax" and instead pin down exactly which geometric constraint governs capacity.'
    prior_limits: 'Delta-rule recurrences like Gated DeltaNet improve recall via gating that erases stale content at repeated addresses, but no theory specifies what capacity ceiling that improvement runs into. Prior work framing recall as a spherical-packing problem exists, but rarely gets implemented in real hardware kernels and compared empirically side by side with softmax and modern linear-attention baselines.'
    goal: 'Derive feature maps from first principles using the geometric structure (symmetric cones) that certifies nonnegative attention weights, precisely quantify the recall-capacity ceiling that geometry allows, and find a practical way to enlarge state just above that ceiling without adding parameters.'
    method: '**KATA** (Kernelized Linear Attention Activations) derives a family of feature maps by certifying nonnegative attention weights through a self-dual homogeneous cone. Rank-one PSD (positive semi-definite) features are shown to have a favorable capacity-interference trade-off, a parameter-free convex output gate is recovered, and recall capacity is characterized via a Welch-bound interference floor. It is akin to packing as many labeled items as possible into limited shelf space without the labels blurring together — pack the items (keys) too tightly and the labels (values) start to interfere.'
    key_idea: 'The core insight is that reframing recall as a "spherical packing" problem unifies softmax''s exponential sharpening and linear attention''s approximate kernels under one geometric account. Above the Welch-bound interference floor, the state can be enlarged **with no added parameters**, because spherical codes exist that pack exponentially many keys into the projection dimension.'
    validation: 'Recall ability is tested via long-range MQAR (multi-query associative recall) and repeated-key-overwrite synthetic tasks; contextual fluency via NIAH, FDA, and SQuAD; language-modeling quality via 340M-parameter pretraining (word-level perplexity, zero-shot accuracy). Kernels are benchmarked at two operating points — a flash-attention-style forward, and an exact O(T) chunked-state form.'
    results: 'Several KATA variants **beat Gated DeltaNet** on long-range MQAR and repeated-key overwrite, retaining **0.985 MQAR at 16x** out-of-distribution length extrapolation — approaching softmax attention with roughly **1/4** the KV-cache entries. Kernels reach **~1.6x** FlashAttention-2 throughput in the flash-style forward and **~11x** at 131k tokens in the chunked-state form. The associative scan lowers inter-chunk recurrence depth to O(log(T/C)) for chunk size C, averaging **2.4x** the throughput of a matched sequential linear-attention baseline.'
    comparison: 'KATA-M1 has the larger geometric packing budget (a 266,240-entry state) yet KATA-M2 obtains stronger hard-needle recall with only 135,168 entries — the model learns two 32-dimensional factors more effectively than one 64-dimensional factor. In other words, the geometric capacity ceiling and the capacity actually realized by training are distinct, and the paper explicitly separates the two.'
    significance: 'From an efficient-architecture standpoint, treating recall failure as a named physical quantity (an interference floor) rather than "approximation error vs. softmax" makes it possible to predict which feature maps will have better capacity, and why. Offers both a theoretical and a practical tool for closing the recall gap that linear/hybrid attention must close to genuinely compete with softmax.'
    limitations: 'The paper''s own "Discussion and Limitations" section states: KATA-M1 is strong on NIAH and direct-copy SWDE but degrades on tasks like FDA and SQuAD that require context interpretation and selection among correlated candidates (KATA-M2 and DeltaKATA-M1 recover part of that gap). The authors also state that the geometric capacity ceiling is an "achievable ceiling" only under the specific real, unit-direction, degree-two model assumed.'
    future_work: 'The paper names four explicit directions: (1) extend packing theory and constructive dictionaries to complex spaces, where spherical codes and ETFs may offer more capacity at equal dimension; (2) systematically pretrain GatedKATA and GatedDeltaKATA and evaluate DeltaKATA beyond M1 (all requiring dedicated fused GPU kernels); (3) test the learnability of the full KATA-M1 geometry beyond d_head=64, with kernels making its quadratic feature state practical at wider heads; (4) explore tensor-train factorizations for higher even-order outer-product features without materializing the full tensor state.'
    resources: 'An official GitHub repository is verified (ayghri/kata, confirmed via curl, HTTP 200) — the code is publicly released.'
thread:
  ko: |-
    선형 어텐션의 회상 문제는 오래된 계보를 갖는다 — 초기 커널 근사들은 소프트맥스를 값싸게 흉내 내려다 회상 능력을 잃었고, Mamba·Gated DeltaNet 같은 순환-상태 아키텍처는 게이팅과 델타-룰로 오래된 내용을 지우며 실증적으로 회상을 개선해왔다. 하지만 "어디까지 개선 가능한가"를 말해주는 이론은 없었다 — 각 아키텍처가 자기만의 트릭으로 경험적 개선치를 보고할 뿐이었다.

    KATA의 개념적 전환은 회상을 스펠컬 패킹 문제로 다시 쓰는 데 있다. 소프트맥스가 암묵적으로 갖는 "어텐션 가중치는 비음수여야 한다"는 제약을 대칭 원뿔이라는 더 넓은 기하학적 틀로 일반화하면, 그 틀 안에서 rank-one PSD 피처가 용량-간섭 트레이드오프의 최적점 근처에 있다는 것을 보일 수 있다. 이로써 "이 피처맵이 왜 저 피처맵보다 나은가"라는 질문이 임시방편적 실험 결과가 아니라 이름 붙은 물리량(Welch bound)에 대한 답이 된다.

    이 논문이 여는 다음 질문은 두 갈래다. 하나는 논문이 직접 제시한 대로 복소 공간·더 높은 차수 피처로의 확장이고, 다른 하나는 논문이 다루지 않는 것 — 이 용량-간섭 기하가 양자화나 저정밀 서빙 같은 다른 효율화 축과 어떻게 상호작용하는가이다. 회상 용량과 양자화 강건성이 같은 근본 기하에 뿌리를 두고 있을 가능성은, 지금은 서로 다른 두 문헌이 따로 다루고 있는 질문이다.
  en: |-
    Linear attention's recall problem has a long lineage — early kernel approximations traded away recall capacity to cheaply mimic softmax, while recurrent-state architectures like Mamba and Gated DeltaNet have empirically improved recall through gating and the delta rule that erase stale content. But no theory said how far that improvement could go — each architecture just reported its own empirical gain from its own trick.

    KATA's conceptual shift is to rewrite recall as a spherical-packing problem. Generalizing softmax's implicit constraint that "attention weights must be nonnegative" to the broader geometric framework of symmetric cones lets you show, within that framework, that rank-one PSD features sit near the optimum of the capacity-interference trade-off. That turns "why is this feature map better than that one" from an ad hoc experimental result into an answer about a named physical quantity — the Welch bound.

    This paper leaves two kinds of open question. One is the extension the paper names directly — complex spaces, higher-order features. The other is what it doesn't touch at all: how this capacity-interference geometry interacts with other efficiency axes like quantization or low-precision serving. Whether recall capacity and quantization robustness are rooted in the same underlying geometry is, for now, a question two separate literatures are asking independently.
sparks:
  - ko: '저자들이 명시한 첫 future-work 방향 — 패킹 이론을 복소 공간으로 확장하면 같은 차원에서 스펠컬 코드·ETF가 더 큰 용량을 줄 수 있다는데, 이것이 실제 학습 가능성(learnability)에서도 이득으로 이어질지는 아직 미검증이다.'
    en: "The authors' own first future-work direction — extending packing theory to complex spaces, where spherical codes and ETFs may offer more capacity at equal dimension — is untested for whether that theoretical gain actually translates into learnability."
  - ko: '논문이 스스로 밝힌 한계 — KATA-M1은 문맥 해석이 필요한 FDA·SQuAD 같은 태스크에서 약하다. 순수 회상 용량과 문맥적 선택 능력이 왜 분리되는지, 그 분리가 기하 자체의 속성인지 학습 과정의 속성인지는 열려 있다.'
    en: "A limitation the paper states itself — KATA-M1 is weak on context-interpretation tasks like FDA and SQuAD. Why pure recall capacity and contextual selection ability come apart, and whether that split is a property of the geometry itself or of the learning process, remains open."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
