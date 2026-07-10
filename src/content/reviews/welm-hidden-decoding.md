---
title: "Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models"
arxivId: "2607.08186"
authors: "Aiwei Liu, Cheng Shi, Chuhan Wu, et al. (48 authors, WeLM team)"
date: 2026-07-10
tags: ["looped-transformer", "scaling", "architecture", "kv-cache"]
topic: 'architecture'
summary: "A continued-pretraining method lets an existing LLM backbone keep improving by expanding each token into several parallel streams with their own KV cache instead of looping the same layers, avoiding the pipeline-parallelism problems that make depth-recurrent transformers hard to scale."
summary_ko: "기존 LLM backbone을 그대로 둔 채, 같은 레이어를 반복하는 대신 각 토큰을 독자적인 KV 캐시를 가진 여러 병렬 스트림으로 확장하는 continued-pretraining 방법으로, depth-recurrent transformer의 스케일링을 어렵게 만드는 pipeline-parallelism 문제를 피해간다."
links: ["lotus"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.08186' }
analysis:
  ko:
    background: 'LLM 스케일링은 주로 Transformer backbone 자체를 키우는 방식으로 이루어져 왔는데, 이미 강력한 모델을 더 키우려면 비용이 큰 사전학습을 다시 해야 한다. Depth-recurrent(looped) transformer는 backbone을 고정한 채 토큰당 연산을 늘리는 대안이지만, 같은 레이어를 반복하는 구조가 대형 모델 학습에 쓰이는 pipeline parallelism과 자연스럽게 맞지 않는다.'
    problem: 'Looped 연산은 파이프라인 병렬화 방식과 맞지 않아 대규모로 스케일링하기 어렵고, 이 때문에 depth-recurrent transformer의 시연은 대체로 소규모에 머물러 왔다.'
    prior_limits: '논문 내 명시 없음 — abstract 수준에서는 기존 looped-transformer 연구들이 정확히 어느 스케일까지 시연되었는지 구체적으로 나열하지 않는다.'
    goal: 'Backbone은 고정한 채로, 표준적인 대형 모델 학습(pipeline parallelism)과 호환되는 방식으로 토큰당 연산을 늘려 이미 학습된 모델의 성능을 계속 향상시키는 것.'
    method: '**Hidden Decoding** — continued pretraining(CPT) 중에 각 토큰을 독립적인 embedding table을 가진 **n개의 스트림**으로 확장하고, 중간 스트림들의 key-value 캐시를 이후 계산의 context로 유지해 Transformer 레이어를 늘리거나 넓히지 않고도 토큰당 내부 연산을 늘린다. 이를 감당 가능하게 만드는 것이 **Stream-Factorized Attention**으로, 대부분의 레이어는 자기 스트림 내부만 attend하고 일부 레이어만 스트림 간 정보를 섞어, attention 비용을 n에 대해 이차가 아니라 대략 선형으로 유지한다.'
    key_idea: '"토큰당 연산을 늘린다"는 목표를 depth 반복이 아니라 **시퀀스 길이 확장**으로 재구성함으로써, 추가 연산이 단지 "더 긴 입력"이 되게 만들어 표준 파이프라인 병렬 학습과 그대로 호환되게 한다.'
    validation: 'n=4로 WeLM-HD4-80B와 WeLM-HD4-617B(100B+ MoE 규모)를 continued pretraining으로 학습해, 동일 backbone의 HD 미적용(non-HD) baseline과 직접 비교했다. 또한 확장 계수 n을 바꿔가며 경향성도 확인했다.'
    results: 'WeLM-HD4-80B와 WeLM-HD4-617B 모두 동일 backbone의 non-HD baseline 대비 성능이 향상되었으며, 저자들은 이를 "100B+ MoE 규모에서 시연된 최초의 시퀀스 길이 스케일링 방법"이라 명시한다. 확장 계수 n이 커질수록 이득이 커지는 경향도 보고한다.'
    comparison: '주된 비교 대상은 동일 backbone의 HD 미적용 continued-pretraining baseline이며, 다른 looped-transformer 방법이나 KV 캐시 압축 기법과의 직접 비교는 abstract 수준에서 보고되지 않는다.'
    significance: '이미 학습된 backbone을 재학습 없이 계속 개선할 수 있는 스케일링 경로를 제공한다는 점에서, 대형 모델을 처음부터 다시 학습하지 않고도 성능을 끌어올리는 효율적 스케일링 관점에서 의미가 있다(리뷰어 판단).'
    limitations: 'Abstract에는 정확한 벤치마크별 수치 표가 제시되어 있지 않다(논문 내 명시 없음). 또한 스트림을 n개로 확장하면 동일한 출력 길이에 대해 KV 캐시가 대략 n배(및 cross-stream 레이어의 추가 비용)로 늘어날 수 있는데, 이 메모리 비용에 대한 정량적 논의는 abstract에서 확인되지 않았다(리뷰어 판단: 순수 연산량 관점의 스케일링 이득이 메모리 비용과 어떻게 상쇄되는지는 본문 확인이 필요하다).'
    future_work: '논문 내 명시 없음 — abstract에는 향후 연구 방향이 제시되지 않는다.'
    resources: '공개된 코드나 모델 가중치 링크는 확인되지 않았다 (공개 링크 확인 안 됨).'
  en:
    background: 'LLM scaling has mainly come from enlarging the Transformer backbone, which requires another costly pretraining run for an already-strong model. Depth-recurrent (looped) transformers are an alternative that keeps the backbone fixed while allocating more computation per token, but looping the same layers does not fit naturally with the pipeline parallelism used to train the largest models.'
    problem: 'Because looped computation is incompatible with standard pipeline parallelism, depth-recurrent transformers are hard to scale, and demonstrations of the approach have generally stayed at small scale.'
    prior_limits: 'Not stated in the paper — the abstract does not enumerate exactly what scale prior looped-transformer work has reached.'
    goal: 'Keep an existing backbone fixed while allocating more per-token computation in a way that is compatible with standard large-model (pipeline-parallel) training, so an already-trained model can keep improving.'
    method: '**Hidden Decoding** expands each token into **n streams** with independent embedding tables during continued pretraining (CPT), keeping the intermediate streams'' key-value cache as context for later computation — adding internal computation per token without adding or widening Transformer layers. This is made affordable by **Stream-Factorized Attention**, where most layers attend only within their own stream and only a few layers mix across streams, keeping the added attention cost roughly linear rather than quadratic in n.'
    key_idea: 'Reframing "more computation per token" as sequence-length expansion rather than depth recurrence means the extra computation is simply a longer input, so it stays directly compatible with standard pipeline-parallel training.'
    validation: 'Trained WeLM-HD4-80B and WeLM-HD4-617B (100B+ MoE scale) at n=4 via continued pretraining, compared directly against matched non-HD baselines of the same backbone, and checked the trend across different expansion factors n.'
    results: 'Both WeLM-HD4-80B and WeLM-HD4-617B improve over their matched non-HD baselines; the authors state this is "the first demonstrated sequence-length scaling method at the 100B+ MoE scale," with gains growing as the expansion factor n increases.'
    comparison: 'The primary comparison is against matched non-HD continued-pretraining baselines of the same backbone; no direct comparison against other looped-transformer methods or KV-cache compression techniques is reported at the abstract level.'
    significance: 'Providing a scaling path that keeps improving an already-trained backbone without a full retrain is relevant to efficient scaling — squeezing more capability out of an existing model rather than paying for a new pretraining run (reviewer judgment).'
    limitations: 'No per-benchmark numeric table is given in the abstract (not stated in the paper). Expanding to n streams could roughly multiply the KV-cache footprint by n (plus the cross-stream layers'' added cost) for the same output length, and no quantitative memory-cost accounting for this was found in the abstract (reviewer judgment: it is unclear from the abstract how the compute-scaling gain nets out against this memory cost).'
    future_work: 'Not stated in the paper — the abstract does not describe future work directions.'
    resources: 'No public code or model weight release could be verified (no public release verified).'
source: "autosweep"
---

## Notes

The most interesting open question this raises (reviewer judgment, not the paper's own framing) is whether Hidden Decoding is actually a compute-efficient/memory-expensive tradeoff — carrying n streams' worth of KV cache for extra per-token computation — the mirror image of the usual acceleration tradeoff of trading memory for compute. The abstract frames the contribution entirely in terms of pipeline-parallelism compatibility and does not discuss this memory cost.
