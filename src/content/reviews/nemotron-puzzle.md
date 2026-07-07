---
title: "Nemotron-Labs-3-Puzzle-75B-A9B: Compressing Hybrid MoE LLMs"
arxivId: "2607.04371"
authors: "Akhiad Bercovich, Talor Abramovich, Daniel Afrimi, Shay Aharon, et al."
lab: "NVIDIA"
date: 2026-07-07
tags: ["compression", "architecture", "hybrid-architecture"]
topic: 'compression'
summary: "An iterative pruning-plus-distillation pipeline compresses a hybrid Mamba/MoE model from 120.7B to 75.3B total parameters, roughly doubling server throughput at matched user-throughput constraints while retaining most downstream accuracy."
summary_ko: "반복적 pruning과 distillation 파이프라인으로 하이브리드 Mamba/MoE 모델을 총 파라미터 1207억에서 753억으로 압축해, 동일한 사용자 처리량 제약에서 서버 처리량을 약 2배로 늘리면서 대부분의 다운스트림 정확도를 유지한다."
links: ["gated-deltanet"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.04371' }
  - { label: 'Model-Optimizer toolkit (GitHub)', url: 'https://github.com/NVIDIA/Model-Optimizer' }
analysis:
  ko:
    background: 'Attention과 Mamba state-space layer를 결합한 대형 hybrid Mixture-of-Experts(MoE) 모델은 높은 정확도를 제공하지만, 그 크기와 메모리 사용량이 interactive 배포 처리량을 제한한다.'
    problem: 'Nemotron-3-Super 같은 매우 큰 hybrid MoE 모델을 엄격한 사용자별 지연 제약 하에서 배포하면, 특히 매우 긴 문맥 길이에서 서버가 동시에 처리할 수 있는 사용자 수가 제한된다.'
    prior_limits: '대형 hybrid MoE/Mamba 모델에 대한 기존 압축 접근법은 보통 단일 축(예: expert pruning만, 또는 quantization만)만 최적화할 뿐, MoE 구조·활성 파라미터 예산·Mamba state 크기를 함께 최적화하지 않는다.'
    goal: 'Interactive 배포 제약 하에서 서버 처리량을 최대화하면서, 부모 모델의 다운스트림 정확도를 최대한 유지하는 훨씬 작은 hybrid MoE 모델을 만드는 것.'
    method: 'Iterative Puzzle 압축 프레임워크는 (이질적(heterogeneous) MoE expert pruning, 활성 파라미터 예산, Mamba state 크기 pruning을 함께 최적화하는) 완만한 pruning 단계와 knowledge-distillation 복구 단계를 번갈아 수행하고, 이어서 강화학습 복구와 Multi-Token Prediction(MTP) head 추가를 거치는 3단계 pruning+distillation 라운드로 구성된다.'
    key_idea: 'MoE expert와 Mamba state를 균일하게 pruning하는 대신, 계층마다 이질적이고 불균일하게 pruning하고(예: Mamba state 채널을 128에서 96으로, 즉 원래의 75%로 축소), 한 번의 압축이 아니라 pruning 라운드 사이사이에 단계적 knowledge distillation으로 품질을 복구한다.'
    validation: '추론(AIME25, GPQA, LiveCodeBench), 장문맥(RULER 256K/512K/1M token), 다국어(MMLU-ProX, WMT24++) 벤치마크를 BF16과 NVFP4 정밀도 모두에서 평가했고, 8xB200 노드에서 처리량을, 단일 H100에서 1M-token 동시성을 측정했다.'
    results: '총 파라미터는 1207억에서 753억(원본의 **62.4%**)으로, 활성 파라미터는 128억에서 93억(**73.1%**)으로 줄어들며, 동일한 사용자 처리량 제약에서 서버 처리량은 Nemotron-3-Super 대비 약 **2배**가 되고, 단일 H100에서 1M-token 동시 처리는 1개 요청에서 **8개 요청**으로 늘어난다. 정확도는 부모 모델에 근접해, 예를 들어 AIME25 89.7% vs 92.2%, RULER-1M 92.2% vs 93.9%다.'
    comparison: 'BF16과 NVFP4 정밀도 모두에서 부모 모델인 Nemotron-3-Super와 직접 비교했으며, Multi-Token Prediction(MTP) 유무로도 자체 비교했다(지속적인 MTP 학습으로 수락 길이가 3.31에서 4.31로 상승).'
    significance: 'MoE expert·활성 파라미터 예산·Mamba state 크기에 대한 단일 축이 아닌 결합 pruning을 통해, 대형 실서비스급 hybrid MoE+Mamba 모델의 총 파라미터를 3분의 1 이상 줄이면서도 대부분의 다운스트림 능력을 보존할 수 있음을 보여준다.'
    limitations: '논문은 가장 큰 정확도 격차가 일부 instruction-following 및 agentic 평가, 구체적으로 Arena-Hard-V2와 특정 TauBench 도메인에서 나타나며, 이들이 공격적인 압축에 더 민감하다고 명시한다.'
    future_work: '이번 리뷰에서는 원문에서 명시적인 future work 절을 확인하지 못했다(리뷰어 판단: 결합 pruning 레시피를 더 큰 활성 파라미터 예산이나 Mamba가 아닌 hybrid layer로 확장하는 것이 자연스러운 다음 단계일 것이다).'
    resources: '압축 방법론은 NVIDIA Model-Optimizer GitHub 저장소에서 언급되는 Puzzletron 툴킷을 기반으로 하며, 이번 리뷰에서는 별도의 모델/체크포인트 공개 링크는 확인하지 못했다.'
  en:
    background: 'Large hybrid Mixture-of-Experts (MoE) models that combine attention with Mamba state-space layers offer strong accuracy, but their size and memory footprint limit interactive deployment throughput.'
    problem: 'Deploying a very large hybrid MoE model like Nemotron-3-Super under strict per-user latency constraints limits how many concurrent users a server can serve, especially at very long context lengths.'
    prior_limits: 'Prior compression approaches to large hybrid MoE/Mamba models typically optimize a single axis (e.g. only expert pruning, or only quantization) rather than jointly optimizing MoE structure, active parameter budget, and Mamba state size together.'
    goal: 'Produce a substantially smaller hybrid MoE model that maximizes server throughput under interactive deployment constraints while retaining as much of the downstream accuracy of the parent model as possible.'
    method: 'The Iterative Puzzle compression framework alternates moderate pruning phases (jointly optimizing heterogeneous MoE expert pruning, active parameter budget, and Mamba state-size pruning) with knowledge-distillation recovery phases, followed by reinforcement-learning recovery and a Multi-Token Prediction (MTP) head, across three staged rounds of pruning plus distillation.'
    key_idea: 'Rather than pruning MoE experts and Mamba state uniformly, the framework prunes each heterogeneously and non-uniformly across layers (e.g. Mamba state channels reduced from 128 to 96, i.e. to 75% of the original size) and recovers quality with staged knowledge distillation between pruning rounds instead of a single one-shot compression step.'
    validation: 'Evaluated on reasoning (AIME25, GPQA, LiveCodeBench), long-context (RULER at 256K/512K/1M tokens), and multilingual (MMLU-ProX, WMT24++) benchmarks in both BF16 and NVFP4 precision, plus throughput measured on 8xB200 nodes and 1M-token concurrency on a single H100.'
    results: 'Total parameters drop from 120.7B to 75.3B (**62.4%** of the original) and active parameters from 12.8B to 9.3B (**73.1%**), while server throughput roughly doubles versus Nemotron-3-Super at matched user-throughput constraints, and 1M-token concurrency on a single H100 rises from 1 request to **8 requests**; accuracy stays close to the parent model, e.g. AIME25 89.7% vs. 92.2%, RULER-1M 92.2% vs. 93.9%.'
    comparison: 'Compared directly against the parent model Nemotron-3-Super across BF16 and NVFP4 precision, and against itself with and without Multi-Token Prediction (MTP acceptance length rises from 3.31 to 4.31 with continued MTP training).'
    significance: 'It demonstrates that a large hybrid MoE+Mamba production model can be compressed by more than a third in total parameters while preserving most downstream capability, using joint (not single-axis) pruning across MoE experts, active parameter budget, and Mamba state size.'
    limitations: 'The paper states the largest accuracy gaps appear on some instruction-following and agentic evaluations, specifically Arena-Hard-V2 and certain TauBench domains, which it describes as more sensitive to aggressive compression.'
    future_work: 'No explicit future-work section was confirmed from the available text for this review (reviewer judgment: extending the joint-pruning recipe to even larger active-parameter budgets or to non-Mamba hybrid layers would be a natural next step).'
    resources: 'The compression methodology builds on the Puzzletron toolkit referenced under the NVIDIA Model-Optimizer GitHub repository; no direct model or checkpoint release link was confirmed from the available text for this review.'
source: "autosweep"
---

## Notes

The headline efficiency number (roughly 2x server throughput at matched user-throughput) comes from jointly pruning three different structural axes at once (MoE experts, active-parameter budget, Mamba state size) rather than optimizing any one of them in isolation, recovered with staged distillation between rounds rather than a single compression pass. The honestly-reported accuracy gaps concentrate on agentic/instruction-following evals, which is a useful signal about where aggressive hybrid-model compression is currently weakest.
