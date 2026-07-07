---
title: "LOTUS: Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers"
arxivId: "2606.31779"
authors: "Ying Fan, Anej Svete, Kangwook Lee"
date: 2026-07-07
tags: ["architecture", "reasoning", "efficiency"]
topic: 'architecture'
summary: "A looped Transformer trained with dense per-latent-position chain-of-thought supervision closes the long-standing accuracy gap between latent and explicit reasoning at 3B parameters while cutting reasoning latency by 2.1-6.9x."
summary_ko: "잠재 위치마다 조밀한 chain-of-thought 지도학습을 적용한 looped Transformer가 30억 파라미터 규모에서 잠재적 추론과 명시적 추론 간의 오랜 정확도 격차를 좁히면서 추론 지연을 2.1-6.9배 단축한다."
links: []
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2606.31779' }
  - { label: 'GitHub', url: 'https://github.com/yingfan-bot/lotus' }
analysis:
  ko:
    background: '명시적 chain-of-thought(CoT)는 중간 추론 단계를 token 단위로 생성하는 반면, 잠재적(latent) CoT는 디코딩된 token 대신 연속적인 표현으로 대체해 모델의 hidden state 내부에서 다단계 추론을 수행함으로써 효율을 높인다.'
    problem: '기존 latent-CoT 방법들은 모델 규모가 대략 10억 파라미터를 넘어서면 명시적 CoT보다 성능이 떨어지며, 이 정확도 격차는 규모가 커질수록 더 벌어진다.'
    prior_limits: 'CODI, SIM-CoT 같은 기존 latent-CoT 접근법은 주로 최종 답 손실만으로 추론을 지도하며 모든 잠재 위치에 조밀한 단계별 지도를 적용하지 않아, 어떤 잠재가 어떤 추론 단계를 나타내야 하는지에 대한 명확한 신호가 없다.'
    goal: '잠재 CoT의 추론 속도 이점을 유지하면서 실제 모델 규모(30억 파라미터)에서 잠재 CoT와 명시적 CoT 간의 정확도 격차를 좁히는 것.'
    method: 'LOTUS는 파라미터를 추가하지 않고 R번의 반복에 걸쳐 가중치를 재사용하는 looped(recurrent-depth) Transformer로, 반복마다 K개의 병렬 잠재 block을 처리하며, 각 잠재 위치에 대응하는 gold chain-of-thought token에 대한 cross-entropy 손실을 모든 잠재 위치에 적용하고 최종 답 손실도 함께 사용해 학습한다.'
    key_idea: '각 잠재 block이 나타내야 할 추론 단계를 알려주는 위치별 지도와, 전체 block 사슬이 함께 정답을 내도록 강제하는 최종 답 손실을 결합하는 것이 잠재 공간을 정확하면서도 해석 가능하게 만드는 핵심이며, ablation은 둘 중 하나만으로는 크게 부족함을 보인다.'
    validation: 'GSM8k-Aug(38.5만 개)로 학습하고 GSM8K(1,319개 테스트) 및 out-of-domain 세트(GSM-Hard, MultiArith, SVAMP)에서 평가했으며, Llama-3.2-3B-Instruct·Llama-1B·GPT-2 백본을 사용하고 설정당 3개 시드를 보고한다.'
    results: '30억 규모에서 LOTUS는 in-domain 70.0%로 명시적 CoT 71.5%에 근접(**1.5점** 차이)하고 out-of-domain에서는 63.9% vs 62.1%로 오히려 앞서며, 이는 기존 latent-CoT 베이스라인(CODI+SIM-CoT) 62.3%(명시적 CoT 대비 약 **9점** 차이)와 대비된다. 정확도가 같은 수준에서 사고 단계 지연은 간결한 수식 표현에서 **2.5배**, 장황한 자연어 CoT에서 **6.9배** 줄어든다.'
    comparison: '동일한 모델 규모에서 명시적 CoT와 기존 latent-CoT 방법(CODI, SIM-CoT)에 대해 비교했으며, ablation에서 단계별 손실이나 최종 답 손실 중 하나를 제거하면 in-domain top-1 gold-step 복원율이 70.9%에서 9.1-9.4%로 붕괴한다.'
    significance: '30억 규모에서 명시적 CoT 정확도에 도달한 최초의 latent-CoT 방법으로 보고되며, loop 이후의 잠재 표현을 base 언어모델 head에 통과시키면 gold 추론 단계(top-1 70.9%, top-5 85.8%)뿐 아니라 학습 중 본 적 없는 타당한 추론 사슬(top-1 15.3%, top-5 64.0%)까지 복원됨을 보여, 잠재 공간이 암기가 아니라 일반화된 표현임을 시사한다.'
    limitations: '평가가 수학 문장제 벤치마크(GSM8K 계열)에 집중되어 있고, 잠재 block 수(K)·block당 token 수(c)·loop 반복 수(R)가 적응적이지 않은 고정된 hyperparameter다(리뷰어 판단: 추론 사슬 길이가 더 가변적인 비수학 추론 영역에서의 성능은 검증되지 않았다).'
    future_work: '저자들은 K·c·R의 적응적 선택을 향후 방향으로 제시한다.'
    resources: '코드는 GitHub의 yingfan-bot/lotus에 공개되어 있다.'
  en:
    background: 'Explicit chain-of-thought (CoT) reasoning generates intermediate steps token-by-token, while latent CoT instead performs multi-step reasoning within hidden states, replacing decoded tokens with continuous representations for greater efficiency.'
    problem: 'Existing latent-CoT methods underperform explicit CoT once model scale passes roughly 1 billion parameters, and this accuracy gap widens further as scale increases.'
    prior_limits: 'Prior latent-CoT approaches (e.g. CODI, SIM-CoT) supervise reasoning mainly through a final-answer loss and do not apply dense, per-step supervision to every latent position, leaving no clear signal for which latent should represent which reasoning step.'
    goal: 'Close the accuracy gap between latent and explicit CoT at a real model scale (3B parameters) while preserving latent CoT inference-speed advantage.'
    method: 'LOTUS is a looped (recurrent-depth) Transformer that reuses its weights across R iterations without adding parameters, processing K parallel latent blocks per iteration, trained with a cross-entropy loss applied to every latent position against the corresponding gold chain-of-thought token, alongside a final-answer loss.'
    key_idea: 'Combining per-position supervision (which tells each latent block what reasoning step it should represent) with a final-answer loss (which forces the whole chain of blocks to be jointly correct) is what makes the latent space both accurate and legible; an ablation shows either loss alone is far from sufficient.'
    validation: 'Trained on GSM8k-Aug (385k examples) and evaluated on GSM8K (1,319 test examples) plus out-of-domain sets (GSM-Hard, MultiArith, SVAMP), using Llama-3.2-3B-Instruct, Llama-1B, and GPT-2 backbones, with 3 seeds reported per configuration.'
    results: 'At 3B scale, LOTUS reaches 70.0% versus explicit CoT 71.5% in-domain (a **1.5-point** gap) and 63.9% versus 62.1% out-of-domain (LOTUS ahead), compared with a prior latent-CoT baseline (CODI+SIM-CoT) at 62.3% (roughly a **9-point** gap to explicit CoT); at matched accuracy, thought-phase latency drops **2.5x** on compact math expressions and **6.9x** on verbose natural-language CoT.'
    comparison: 'Compared against explicit CoT and prior latent-CoT methods (CODI, SIM-CoT) at matched model scale; an ablation shows removing either the per-step or the final-answer loss collapses in-domain top-1 gold-step recovery from 70.9% to 9.1-9.4%.'
    significance: 'It is reported as the first latent-CoT method to match explicit CoT accuracy at 3B scale, and shows that projecting the post-loop latents through the base language-model head recovers gold reasoning steps (70.9% top-1, 85.8% top-5) and even previously-unseen valid reasoning chains (15.3% top-1, 64.0% top-5), suggesting the latent space generalizes rather than memorizes.'
    limitations: 'Evaluation is concentrated on math word-problem benchmarks (the GSM8K family), and the number of latent blocks (K), tokens per block (c), and loop iterations (R) are fixed, hand-set hyperparameters rather than adaptive ones (reviewer judgment: performance on non-math reasoning domains with more variable reasoning-chain length is untested).'
    future_work: 'The authors name adaptive selection of K, c, and R as a direction for future work.'
    resources: 'Code is released on GitHub at yingfan-bot/lotus.'
source: "autosweep"
---

## Notes

The ablation table is the load-bearing evidence here: neither the per-step latent supervision nor the final-answer loss alone gets anywhere close to the combined result, which suggests the fix is less a new architectural primitive and more a credit-assignment trick — give every latent slot a specific target and check the whole chain is jointly correct. Worth flagging that every reported number is on math word problems; whether the same recipe holds for reasoning tasks with more variable chain length and structure is untested.
