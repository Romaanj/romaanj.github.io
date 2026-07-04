---
title: 'H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs'
arxivId: '2306.14048'
authors: 'Zhang et al.'
lab: 'UT Austin / Stanford / CMU'
venue: 'NeurIPS 2023'
date: 2026-06-30
tags: [kv-cache, sparse-attention, serving]
topic: kv-cache
summary: 'A small set of "heavy-hitter" tokens dominates attention mass, so greedily evicting low-accumulated-attention entries keeps only ~20% of the KV cache while matching full-cache accuracy and raising serving throughput up to 29x.'
summary_ko: 'Attention mass의 대부분을 소수 "heavy-hitter" 토큰이 차지한다는 관측을 바탕으로, 누적 attention score가 낮은 엔트리를 greedy하게 evict해 KV cache의 약 20%만 유지하면서 full-cache 정확도를 유지하고 serving throughput을 최대 29배 높인 eviction 정책.'
links: [kivi, kvquant, duo-attention]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2306.14048' }
  - { label: 'PDF', url: 'https://arxiv.org/pdf/2306.14048' }
  - { label: 'GitHub', url: 'https://github.com/FMInference/H2O' }
analysis:
  ko:
    background: 'LLM generative inference에서 KV cache는 sequence 길이와 batch size에 비례해 GPU 메모리를 선형으로 잠식하며, long-content 생성(dialogue, story writing)에서는 model weight보다 cache가 배포 비용의 병목이 된다.'
    problem: 'Full KV cache를 유지하면 batch size와 context 길이가 메모리에 묶여 throughput이 제한되는데, 어떤 토큰의 K/V를 버려도 생성 품질이 유지되는지에 대한 원칙적 eviction 기준이 없었다.'
    prior_limits: 'Sparse Transformer류의 고정 패턴(strided/fixed)이나 최근 토큰만 남기는 "local" 정책은 attention mass를 실제로 받는 토큰을 함께 버려서 낮은 budget에서 정확도가 급락한다.'
    goal: '재학습·fine-tuning 없이, 고정된 cache budget(예: **20%**)만 유지하면서 full-cache 정확도를 보존하고 serving throughput을 높이는 online KV eviction 정책을 만드는 것.'
    method: '각 decoding step에서 past-token attention score를 토큰별로 누적하고, budget k 초과 시 누적 점수가 가장 낮은 엔트리를 greedy하게 evict하되 최근 토큰은 항상 보존해 retained set = recent + heavy hitters로 유지하며, 이를 dynamic submodular maximization으로 정식화해 mild assumption 하에 greedy의 near-optimality 보장을 증명했다.'
    key_idea: 'Densely trained LLM도 attention matrix는 **95%** 이상 sparse하고 누적 attention score가 power-law를 따르므로, 소수 heavy-hitter 토큰(텍스트 내 빈번한 co-occurrence와 상관)이 attention value의 대부분을 차지하며 past-only local 통계만으로도 이들을 online 식별할 수 있다.'
    validation: 'OPT-6.7B/30B/66B, LLaMA-7B/13B, GPT-NeoX-20B를 lm-eval-harness/HELM의 8개 task(COPA, MathQA, OpenBookQA, PiQA, RTE, Winogrande, XSUM, CNN/DailyMail)에서 cache budget 4~100%로 평가하고, A100-80GB(정확도)와 T4-16GB(throughput)에서 시스템 성능을 측정했다.'
    results: '**20%** budget에서 full-cache 정확도와 대등(PiQA **79.22** vs 80.09)하고, throughput은 DeepSpeed Zero-Inference·HF Accelerate 대비 최대 **29배**, FlexGen 대비 **3배**, 같은 batch size에서 latency 최대 **1.9배** 감소를 달성했다.'
    comparison: 'Full KV cache, recency-only "local", Sparse Transformer(strided/fixed)를 주 baseline으로 비교하고 appendix에서 SpAtten·StreamingLLM과도 대조했다.'
    significance: 'Heavy hitter·cache budget·eviction policy라는 용어 체계를 정립해 training-free KV eviction을 KV 압축의 독립 축으로 확립했고, entry를 줄이는 이 방향은 entry당 bit를 줄이는 quantization(KIVI/KVQuant)과 직교해 결합 가능하다.'
    limitations: '이론 보장은 submodularity 가정 하에서만 성립하고 논문 자체의 한계 논의는 짧으며, 누적 attention이 backward-looking 통계라서 나중에야 중요해지는 토큰(long-context retrieval)을 미리 evict하는 실패 모드는 후속 연구들이 문서화했다 (리뷰어 판단).'
    future_work: '논문은 quantization 등 직교 최적화와의 결합과 이론 가정의 검증을 후속 방향으로 시사하며, head/layer별 이질성을 반영한 adaptive budget과 retrieval-safe eviction이 자연스러운 다음 단계다 (리뷰어 판단).'
    resources: '공식 코드가 GitHub(FMInference/H2O)에 공개되어 있고 FlexGen 기반 serving 구현을 포함하며, 별도 checkpoint는 없다(training-free).'
  en:
    background: 'In LLM generative inference the KV cache grows linearly with sequence length and batch size, and for long-content generation (dialogue, story writing) it, rather than the model weights, becomes the deployment memory bottleneck.'
    problem: 'Keeping the full KV cache ties batch size and context length to GPU memory and caps throughput, and there was no principled criterion for which tokens K/V entries can be dropped without hurting generation quality.'
    prior_limits: 'Fixed sparsity patterns (Sparse Transformer strided/fixed) and recency-only "local" caches also discard tokens that actually receive attention mass, so accuracy collapses at low budgets.'
    goal: 'Build an online, training-free KV eviction policy that holds a fixed cache budget (e.g., **20%**) while preserving full-cache accuracy and raising serving throughput.'
    method: 'At each decoding step it accumulates every past token attention score, and once the budget k is exceeded it greedily evicts the lowest-accumulated-score entry while always protecting recent tokens (retained set = recent + heavy hitters), formalizing eviction as dynamic submodular maximization with a proven near-optimality guarantee for the greedy policy under mild assumptions.'
    key_idea: 'Even densely trained LLMs have over **95%** sparse attention matrices with power-law accumulated scores, so a few heavy-hitter tokens (correlated with frequent textual co-occurrence) carry most attention value and can be identified online from past-only local statistics.'
    validation: 'Evaluated OPT-6.7B/30B/66B, LLaMA-7B/13B, and GPT-NeoX-20B on eight lm-eval-harness/HELM tasks (COPA, MathQA, OpenBookQA, PiQA, RTE, Winogrande, XSUM, CNN/DailyMail) at cache budgets from 4% to 100%, with accuracy on A100-80GB and throughput on T4-16GB.'
    results: 'At a **20%** budget it matches full-cache accuracy (PiQA **79.22** vs 80.09) and improves throughput by up to **29x** over DeepSpeed Zero-Inference and HF Accelerate and **3x** over FlexGen, cutting latency by up to **1.9x** at the same batch size.'
    comparison: 'Main baselines are the full KV cache, the recency-only "local" policy, and Sparse Transformer (strided/fixed), with appendix comparisons against SpAtten and StreamingLLM.'
    significance: 'It established the heavy-hitter / cache-budget / eviction-policy vocabulary and made training-free KV eviction a standalone axis of KV compression, orthogonal and composable with per-entry quantization (KIVI/KVQuant).'
    limitations: 'The theoretical guarantee holds only under the assumed submodularity and the paper''s own limitations discussion is brief, while the backward-looking accumulated-attention statistic can evict tokens that only become important later (long-context retrieval), a failure mode documented by follow-up work (reviewer judgment).'
    future_work: 'The paper points to composing with orthogonal optimizations such as quantization and validating the theoretical assumptions, with head/layer-adaptive budgets and retrieval-safe eviction as the natural next steps (reviewer judgment).'
    resources: 'Official code is released on GitHub (FMInference/H2O) including a FlexGen-based serving implementation; no checkpoints are needed since the method is training-free.'
source: 'manual'
---

## Notes

H2O is one of the founding papers of KV-cache eviction: it turned the empirical observation of power-law attention skew into an online policy (recent + heavy hitters) with a submodular-greedy justification, and defined the vocabulary the whole subfield still uses. It is the sparsity-side complement to quantization work like KIVI and KVQuant — one drops cache entries, the others shrink them — and its known weakness (backward-looking scores evicting future-relevant tokens) is exactly what head-level approaches like DuoAttention later addressed.

H2O는 KV eviction 계열의 출발점으로, 이후 등장한 거의 모든 budget-based cache 압축 기법이 이 논문의 heavy-hitter 프레임 위에서 정의된다.
