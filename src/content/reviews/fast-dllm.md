---
title: 'Fast-dLLM: Training-Free Acceleration of Diffusion LLMs via KV Cache and Parallel Decoding'
arxivId: '2505.22618'
authors: 'Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, Enze Xie'
lab: 'NVIDIA / MIT / HKU'
date: 2026-07-04
tags: [diffusion-llm, kv-cache, efficiency]
topic: 'diffusion-llm'
summary: 'A block-wise approximate KV cache plus confidence-thresholded parallel decoding accelerates open diffusion LLMs by up to 27.6x end-to-end with 1-2 point accuracy loss, entirely training-free.'
summary_ko: 'Block 단위 approximate KV cache와 confidence threshold 기반 parallel decoding으로, 재학습 없이 open diffusion LLM 추론을 최대 27.6배 가속하면서 정확도 하락을 1-2pt 이내로 유지한다.'
links: [llada, kivi]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2505.22618' }
  - { label: 'GitHub', url: 'https://github.com/NVlabs/Fast-dLLM' }
  - { label: 'Project', url: 'https://nvlabs.github.io/Fast-dLLM/' }
analysis:
  ko:
    background: 'Diffusion LLM(LLaDA, Dream)은 non-autoregressive parallel 생성을 약속하지만, 공개 모델의 실측 추론 속도는 KV cache를 쓰는 AR 모델보다 오히려 느리다.'
    problem: 'Bidirectional attention 때문에 표준 AR KV cache를 쓸 수 없고, 한 step에 여러 token을 naive하게 동시 확정하면 생성 품질이 급락한다.'
    prior_limits: '기존 dLLM 추론은 매 denoising step마다 전체 attention을 재계산했고, 고정 token-per-step 스케줄은 conditional independence 가정 아래 token 간 의존성을 깨뜨린다는 것이 품질 저하의 근본 원인으로 지목된다.'
    goal: '모델 가중치를 건드리지 않는 training-free 방식으로, 정확도 손실을 최소화하며 dLLM 추론 throughput을 AR 수준에 근접시키는 것.'
    method: '생성을 block 단위로 진행하며 prefix(및 DualCache 변형에서는 masked suffix까지)의 KV activation을 캐시해 block 경계에서만 갱신하고, 각 step에서는 max-softmax confidence가 threshold τ(기본 **0.9**)를 넘는 token만 선별적으로 확정하는 confidence-aware parallel decoding(전부 미달 시 최고-confidence token은 항상 확정해 진행 보장)을 결합한다.'
    key_idea: '인접 denoising step 간 block 밖 KV activation은 cosine similarity가 매우 높아 approximate 재사용이 안전하고, Theorem 1은 n개 token의 confidence가 (n+1)·ε ≤ 1을 만족할 만큼 높으면 greedy parallel decoding이 greedy sequential decoding과 일치함을 증명한다.'
    validation: 'LLaDA/LLaDA-1.5/Dream(및 multimodal LLaDA-V)을 GSM8K(5-shot)/MATH(4-shot)/HumanEval/MBPP(3-shot)(+MathVista/MathVerse)에서 생성 길이 256-1024, 단일 A100 80GB, lm-eval 기반으로 평가했다.'
    results: '최대 **27.6×** end-to-end throughput 향상(LLaDA, 8-shot, 1024 token), GSM8K-512에서 **11.0×**, 정확도는 GSM8K-256 기준 79.3→**78.5%**로 1-2pt 이내 하락, 단독 성분은 KV cache **3.2×**·parallel decoding **2.5×**·결합 **8.1×**다.'
    comparison: '가속 없는 vanilla LLaDA/Dream decoding과 고정 top-K token-per-step 베이스라인 대비 평가했으며, 두 성분이 상보적으로 곱해짐(3.2×·2.5× → 8.1×)을 보인다.'
    significance: 'Open dLLM 추론을 재학습 없이 KV-cached AR decoding과 실질적으로 경쟁 가능하게 만든 첫 대표 사례로, cache staleness와 adaptive parallelism을 측정 가능한 조절 변수로 정식화해 이후 dLLM 가속 연구의 표준 베이스라인이 되었다.'
    limitations: '논문 스스로 큰 batch에서는 full attention 오버헤드로 AR LLaMA throughput에 뒤진다고 명시하고 multimodal LLaDA-V는 block size에 민감(작은 block에서 **>8%** 하락)하며, accuracy-staleness frontier의 체계적 지도는 미완이다(리뷰어 판단).'
    future_work: '별도 future work 절은 없으며, per-position adaptive threshold·staleness 인지형 cache refresh·AR용 cache 압축(quantization/eviction)과의 결합이 자연스러운 다음 단계다(리뷰어 판단).'
    resources: '코드와 프로젝트 페이지가 NVlabs GitHub(Fast-dLLM)로 공개되어 있고, training-free 방법이라 별도 checkpoint는 불필요하다.'
  en:
    background: 'Diffusion LLMs (LLaDA, Dream) promise non-autoregressive parallel generation, yet the measured inference speed of open models lags behind KV-cached autoregressive models.'
    problem: 'Bidirectional attention rules out the standard AR KV cache, and naively committing many tokens per step sharply degrades generation quality.'
    prior_limits: 'Prior dLLM inference recomputed full attention at every denoising step, and fixed token-per-step schedules break inter-token dependencies under the conditional independence assumption — identified here as the root cause of parallel-decoding quality loss.'
    goal: 'Bring dLLM inference throughput close to AR levels training-free, without touching model weights and with minimal accuracy loss.'
    method: 'Generation proceeds block by block with prefix KV activations cached and refreshed only at block boundaries (the DualCache variant also caches the masked suffix), combined with confidence-aware parallel decoding that commits only tokens whose max-softmax confidence exceeds a threshold τ (default **0.9**), with a highest-confidence fallback to guarantee progress.'
    key_idea: 'KV activations outside the active block are nearly static across adjacent denoising steps (high cosine similarity), and Theorem 1 proves greedy parallel decoding matches greedy sequential decoding whenever the n committed tokens satisfy (n+1)·ε ≤ 1 in confidence.'
    validation: 'Evaluated LLaDA/LLaDA-1.5/Dream (plus multimodal LLaDA-V) on GSM8K (5-shot), MATH (4-shot), HumanEval, MBPP (3-shot) (+MathVista/MathVerse), generation lengths 256-1024, single A100 80GB, using lm-eval.'
    results: 'Up to **27.6×** end-to-end throughput (LLaDA, 8-shot, 1024 tokens) and **11.0×** on GSM8K-512, with accuracy within 1-2 points (GSM8K-256: 79.3 → **78.5%**); component-wise, KV cache alone gives **3.2×**, parallel decoding **2.5×**, combined **8.1×**.'
    comparison: 'Compared against vanilla LLaDA/Dream decoding and fixed top-K token-per-step baselines, showing the two components compose multiplicatively (3.2× · 2.5× → 8.1×).'
    significance: 'The first widely adopted recipe that makes open dLLM inference practically competitive with KV-cached AR decoding without retraining, formalizing cache staleness and adaptive parallelism as measurable tuning knobs and becoming the standard baseline for subsequent dLLM acceleration work.'
    limitations: 'The paper itself notes that at larger batch sizes full-attention overhead lets AR LLaMA overtake it in throughput, and that multimodal LLaDA-V is block-size sensitive (**>8%** accuracy drop at small blocks); a systematic accuracy-staleness frontier map remains open (reviewer judgment).'
    future_work: 'No formal future-work section; natural next steps are per-position adaptive thresholds, staleness-aware cache refresh, and composing with AR-style cache compression (quantization/eviction) (reviewer judgment).'
    resources: 'Code and project page are released under NVlabs on GitHub (Fast-dLLM); being training-free, no separate checkpoints are required.'
source: 'manual'
---

## Notes

Fast-dLLM is the paper that made open diffusion-LLM inference feel competitive: it identifies the two structural reasons LLaDA and Dream decode slowly — no KV cache under bidirectional attention, and dependency violations under naive parallel commits — and fixes both without any retraining. The block-wise approximate cache turns "how stale can cached states get?" into a tunable block-size knob, while the (n+1)·ε ≤ 1 condition of Theorem 1 gives the confidence-threshold heuristic an actual correctness footing rather than leaving it purely empirical. Its main honestly-stated boundary is batching: at larger batch sizes the full-attention compute floor lets autoregressive models pull back ahead, which is exactly where follow-up work on dLLM serving now concentrates.

LLaDA 계열 dLLM 가속 논문을 읽을 때 사실상의 공통 베이스라인이므로, block size와 confidence threshold 두 knob의 trade-off 곡선을 기준점으로 잡고 후속 논문들을 비교하면 좋다.
