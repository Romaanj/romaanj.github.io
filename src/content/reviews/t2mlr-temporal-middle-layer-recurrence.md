---
title: "T2MLR: Transformer with Temporal Middle-Layer Recurrence"
arxivId: "2607.15178"
date: 2026-07-18
tags: ["architecture", "recurrence", "reasoning"]
topic: 'architecture'
summary: "Fuses a cached middle-layer representation from the previous token directly into an earlier layer of the current token, letting abstract intermediate computation persist across decoding steps for a fraction of the network's depth."
summary_ko: '이전 토큰의 캐싱된 중간 레이어 표현을 현재 토큰의 더 앞쪽 레이어에 직접 융합해, 네트워크 깊이의 일부만으로도 추상적인 중간 연산이 디코딩 스텝을 넘어 지속되게 만드는 구조.'
links: ["deeploop", "layernorm-looped", "lotus"]
resources:
  - label: "arXiv abstract"
    url: "https://arxiv.org/abs/2607.15178"
figures:
  - src: /figures/t2mlr-temporal-middle-layer-recurrence/fig1.png
    caption: "T2MLR fuses a representation from a deep layer at the previous token position into a shallow layer at the current token position, giving better pretraining perplexity and downstream reasoning performance."
    caption_ko: "T2MLR은 이전 토큰 위치의 깊은 레이어 표현을 현재 토큰 위치의 얕은 레이어로 융합해, 더 나은 사전학습 퍼플렉시티와 하위 추론 성능을 얻는다."
    credit: "Figure 1 from arXiv:2607.15178 — authors' figure"
analysis:
  ko:
    background: '표준 트랜스포머는 자기회귀적으로 디코딩하면서, 한 토큰이 여러 레이어를 거쳐 쌓아올린 풍부한 중간 연산(hidden computation)을 매번 토큰 공간(하나의 다음-토큰 확률분포)으로 압축해버린다. 이 압축은 다음 토큰을 예측할 때 이전의 추상적 중간 상태를 그대로 이어받을 방법을 남기지 않는다.'
    problem: '따라서 토큰을 넘나드는 추상적 추론 상태(reasoning state)가 디코딩 스텝 사이에서 지속되기 어렵다 — 각 토큰은 사실상 처음부터 다시 그 추상 상태를 재구성해야 한다.'
    prior_limits: '기존의 루프형(looped)·재귀 깊이(recurrent-depth) 트랜스포머들은 전체 레이어를 통째로 반복(loop over all layers)하는 방식이 일반적이었는데, 이는 연산 비용이 크고 어느 레이어가 실제로 재귀로부터 이득을 보는지 구분하지 않는다.'
    goal: '전체 네트워크를 반복하지 않고도, 이전 토큰의 깊은 레이어 계산 결과를 현재 토큰이 이어받게 해 더 강한 추론 성능을 얻는 것이 목표다.'
    method: 'T2MLR은 이전 토큰 위치의 특정 중간(middle) 레이어 표현을 캐싱해두었다가, 현재 토큰 위치의 더 앞쪽(shallow) 레이어에 직접 융합한다 — 전체 레이어를 반복하는 대신 국소적인 블록(논문에서는 전체 네트워크의 약 **20%**에 해당하는 중간 레이어 구간)에만 타깃으로 재귀를 적용한다.'
    key_idea: '핵심은 "어디를 반복할지"를 고르는 것이다 — 모든 레이어를 도는 대신, 이전 토큰의 깊은 레이어 표현 하나를 현재 토큰의 얕은 레이어에 지름길처럼 연결한다. 마치 매번 처음부터 생각을 다시 정리하는 대신, 방금 도달했던 결론의 요약본을 다음 생각의 출발점으로 바로 건네받는 것과 같다.'
    validation: '자연어 사전학습과 다중 홉(multi-hop) 추론 파인튜닝 과제로 평가했고, 수학 추론 벤치마크(GSM8K, MATH500)와 ARC-C/ARC-E/HellaSwag 등 제로샷 NLP 과제로 검증했다. 기존 1.7B 파라미터 모델에 짧은 파인튜닝만으로 재귀 구조를 이식(retrofit)하는 실험도 포함한다.'
    results: 'GSM8K 정확도가 **35.78% → 39.88%**(상대 **+11.5%**), MATH500이 **12.80% → 18.00%**(상대 **+40.6%**)로 개선되었으며, 제로샷 NLP 과제들에서도 일관되지만 더 작은 폭의 향상을 보였다. 전체 레이어가 아니라 중간 레이어의 약 **20%**만 반복했을 때 가장 좋은 결과가 나왔다.'
    comparison: '데이터·파라미터를 맞춘(data- and parameter-matched) 표준 트랜스포머 베이스라인과 일관되게 비교해 앞선다고 보고한다. 전체 레이어를 반복하는 기존 루프형 트랜스포머 대비, 국소적(20%) 재귀만으로 더 강한 추론 이득을 얻는다는 것이 차별점이다.'
    significance: '루프형·재귀 깊이 트랜스포머 계열 연구에 "어느 레이어를 반복할지"라는 설계 축을 추가한다 — 전체 깊이를 반복하는 것이 재귀의 이득 대부분을 내는 데 필요하지 않을 수 있다는 실증이며, 짧은 파인튜닝만으로 기존 사전학습 모델에 이식 가능하다는 점에서 실용적 의미도 크다.'
    limitations: '학습이 파라미터를 맞춘 표준 트랜스포머 베이스라인보다 느리다 — 재귀 블록 크기에 따라 학습 시 약 **2~4배**의 wall-clock 오버헤드가 발생하며, 저자들은 이를 "트레이드오프"로 명시한다.'
    future_work: '저자들은 (1) 더 나은 시간축 믹싱(temporal mixing) 메커니즘을 통한 학습 비용 절감, (2) 재귀 경로를 통한 정보 흐름의 메커니즘적 이해, (3) 더 큰 모델과 더 어려운 벤치마크로의 확장, (4) 강화학습 환경으로의 응용을 향후 연구 방향으로 제시한다.'
    resources: 'arXiv 프리프린트가 유일하게 확인된 공개 자료다 (리뷰어 판단 — 코드 공개 여부는 확인되지 않음).'
  en:
    background: 'Standard Transformers decode autoregressively while repeatedly compressing rich hidden computation — built up across many layers for one token — down into token space (a single next-token distribution). That compression leaves no route for an abstract intermediate state to carry over intact into the next token.'
    problem: 'As a result, abstract reasoning state that spans multiple tokens struggles to persist across decoding steps — each new token effectively has to reconstruct that abstract state from scratch.'
    prior_limits: 'Prior looped / recurrent-depth Transformers typically loop over the entire stack of layers, which is compute-expensive and does not distinguish which layers actually benefit from the recurrence.'
    goal: 'Get the reasoning benefits of recurrence without looping the whole network, by letting the current token inherit a deep-layer computation result from the previous token.'
    method: 'T2MLR caches a representation from a specific middle layer at the previous token position, then fuses it directly into an earlier (shallower) layer at the current token position — applying recurrence to a localized block rather than the whole stack (the paper finds the best results looping only about **20%** of the middle layers).'
    key_idea: 'The key move is choosing *where* to loop rather than looping everything — one deep-layer representation from the previous token is wired as a shortcut into a shallow layer of the current token. It''s like handing off a one-line summary of the conclusion you just reached as the starting point for your next thought, instead of re-deriving it from scratch every time.'
    validation: 'Evaluated on natural-language pretraining and multi-hop reasoning fine-tuning, tested on math reasoning benchmarks (GSM8K, MATH500) and zero-shot NLP tasks (ARC-C, ARC-E, HellaSwag, etc.), including retrofitting the recurrence into an existing 1.7B-parameter model with brief fine-tuning.'
    results: 'GSM8K accuracy improves **35.78% → 39.88%** (**+11.5%** relative), MATH500 improves **12.80% → 18.00%** (**+40.6%** relative), with consistent but smaller gains on zero-shot NLP tasks. The best results come from looping only about **20%** of the middle layers, not the full stack.'
    comparison: 'Reports consistent gains over data- and parameter-matched standard Transformer baselines. Compared to prior full-stack looped Transformers, the distinguishing claim is that localized (20%) recurrence delivers stronger reasoning gains than looping everything.'
    significance: 'Adds a new design axis — *which* layers to loop, not just whether to loop — to the looped/recurrent-depth Transformer line, with evidence that looping the full depth isn''t necessary to capture most of recurrence''s benefit; also practically notable since it retrofits onto an existing pretrained model with brief fine-tuning.'
    limitations: 'Training is slower than a parameter-matched Transformer baseline — roughly **2-4×** wall-clock training overhead depending on the recurrent block size, which the authors explicitly call out as a trade-off.'
    future_work: 'The authors point to (1) reducing training cost via better temporal-mixing mechanisms, (2) a mechanistic understanding of information flow through the recurrent pathway, (3) scaling to larger models and harder benchmarks, and (4) applications in reinforcement-learning settings.'
    resources: 'The arXiv preprint is the only verified public resource (reviewer judgment — whether code is separately released is not confirmed).'
thread:
  ko: |-
    루프형·재귀 깊이 트랜스포머 계열(Universal Transformer 이후 Ouro, DeepLoop, LoopFormer 등)은 대체로 "전체 레이어를 몇 번 반복할 것인가"라는 단일 축으로 설계돼 왔다. T2MLR은 여기에 두 번째 축 — "레이어 중 어디를 반복할 것인가" — 을 더한다.

    개념적 전환은 재귀를 네트워크 전체의 속성이 아니라 국소적(20% 규모) 블록의 속성으로 재정의한 것이다. 이전 토큰의 깊은 레이어 표현 하나를 현재 토큰의 얕은 레이어로 지름길처럼 연결하는 방식은, 전체를 반복하는 기존 방법보다 적은 오버헤드로 유사하거나 더 나은 추론 이득을 낼 수 있음을 보여준다.

    이 논문이 여는 다음 질문은, 이런 국소적·타깃형 재귀가 이 사이트에서 함께 다루는 다른 루프형 아키텍처(Parcae류의 안정성 조건, LoopFormer의 탄력적 깊이)와 결합될 수 있는지, 그리고 어떤 레이어가 재귀로부터 이득을 보는지를 사전에 예측하는 원리가 존재하는지이다.
  en: |-
    The looped / recurrent-depth Transformer line (Universal Transformer, then Ouro, DeepLoop, LoopFormer, and others) has largely been designed along one axis: how many times to loop the *entire* stack. T2MLR adds a second axis — *which* layers to loop.

    The conceptual shift is redefining recurrence as a property of a localized (~20%) block rather than the whole network. Wiring one deep-layer representation from the previous token as a shortcut into a shallow layer of the current token achieves comparable or stronger reasoning gains than full-stack looping, at lower overhead.

    The question this opens is whether this kind of localized, targeted recurrence composes with the other looped architectures tracked on this site (Parcae-style stability conditions, LoopFormer's elastic depth), and whether there's a general principle for predicting in advance which layers benefit most from recurrence.
sparks:
  - ko: "온라인 강화학습 환경에서 이 국소 재귀가 어떻게 작동하는지는 저자들이 직접 향후 연구로 언급한 방향이다 — 학습 신호가 희소한 RL에서도 같은 지름길 연결이 안정적으로 이득을 주는지는 미확인."
    en: "The authors themselves flag RL settings as future work — whether this localized recurrence remains stable and beneficial under RL's sparser training signal is untested."
  - ko: "저자들이 언급한 '더 나은 시간축 믹싱 메커니즘'이 구체적으로 무엇을 가리키는지는 논문에 명시되지 않았다 — 학습 오버헤드(2~4배)를 줄이는 다른 방법이 있는지는 열린 문제."
    en: "The authors' own future-work item — 'better temporal-mixing mechanisms' to cut training cost — isn't spelled out in the abstract; what would actually reduce the 2-4x training overhead remains open."
  - ko: "재귀 경로를 통한 정보 흐름의 메커니즘적 이해가 아직 없다는 점은 저자들이 명시한 한계다 — 어떤 종류의 추상 정보가 이 지름길로 전달되는지는 해석가능성 연구의 열린 대상이다."
    en: "The authors explicitly name mechanistic understanding of the recurrent information flow as missing — what kind of abstract information actually travels through this shortcut remains open for interpretability work."
source: "autosweep"
---

## Notes
