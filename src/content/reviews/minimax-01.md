---
title: 'MiniMax-01: Scaling Foundation Models with Lightning Attention'
arxivId: '2501.08313'
authors: 'MiniMax (90 authors)'
lab: 'MiniMax'
date: 2026-07-04
tags: [hybrid-architecture, linear-attention, architecture]
topic: 'hybrid-architecture'
summary: 'One softmax-attention block per seven lightning (linear) attention blocks, scaled to a 456B-parameter MoE, matches GPT-4o-class quality while training at 1M-token context and extrapolating to 4M.'
summary_ko: 'Softmax attention 1층 + lightning(linear) attention 7층 hybrid를 456B MoE로 스케일업해, GPT-4o급 품질을 유지하면서 1M-token 학습·4M-token 추론 context를 달성한 최초의 frontier-scale linear attention 모델.'
links: [gated-deltanet, duo-attention, deepseek-r1]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2501.08313' }
  - { label: 'PDF', url: 'https://arxiv.org/pdf/2501.08313' }
  - { label: 'GitHub', url: 'https://github.com/MiniMax-AI/MiniMax-01' }
  - { label: 'HuggingFace', url: 'https://huggingface.co/MiniMaxAI/MiniMax-Text-01' }
analysis:
  ko:
    background: 'Linear attention은 softmax attention의 quadratic 연산량과 계속 자라는 KV cache에 대한 O(n) 대안으로 오래 제시되어 왔지만, 소규모 학술 모델 밖에서는 검증된 적이 없었다.'
    problem: 'Linear attention 위주 아키텍처가 수천억 parameter·백만 token context 규모에서 최신 softmax attention LLM 품질을 따라갈 수 있는지는 검증되지 않은 열린 문제였다.'
    prior_limits: '기존 linear/SSM 계열(TransNormer, Mamba2, HGRN2)은 소규모 학습에 그쳤고 needle-retrieval(NIAH) 능력이 뚜렷하게 약하며, 성숙한 분산 학습·추론 스택은 전부 softmax attention 전용으로 최적화되어 있었다.'
    goal: 'Attention 비용을 near-linear로 유지해 1M-token context를 네이티브로 학습하고 추론에서 4M token까지 extrapolate하는 frontier급 오픈 모델을 만드는 것이 목표다.'
    method: 'MiniMax-Text-01은 456B parameter / 45.9B activated MoE transformer(80 layers, hidden 6144, 32 experts top-2 routing)로, lightning attention(TransNormer linear attention의 I/O-aware tiling 구현) block 7개마다 GQA softmax attention block 1개를 배치하고 head dimension 절반에만 RoPE를 적용하며, LASP+·varlen ring attention·expert-parallel overlap 위에서 3단계 context 확장으로 1M token까지 학습했다.'
    key_idea: 'Softmax attention을 1/8만 섞으면 pure linear attention이 잃는 retrieval 능력이 복원되어, hybrid가 retrieval과 length extrapolation에서 full softmax를 따라잡거나 능가하면서 layer의 7/8을 linear 비용으로 유지한다는 것이 핵심이다.'
    validation: '70M-7B 모델을 최대 300B token으로 학습한 Chinchilla식 scaling law로 softmax/lightning/hybrid를 비교하고, 본 모델은 MMLU/MMLU-Pro/GPQA/IFEval/Arena-Hard, 1M-token까지의 RULER, LongBench-v2, 4M-token NIAH pressure test, MTOB long in-context learning으로 평가했다.'
    results: 'MMLU **88.5**, GPQA-diamond **54.4**를 기록하고, 1M-token 입력에서 RULER **0.910**을 유지하며, LongBench-v2 with-CoT **56.5**로 평가 대상 중 최고, NIAH는 **4M** token까지 near-perfect, 추론은 H20 GPU에서 MFU **75%** 이상을 달성했다.'
    comparison: 'GPT-4o(11-20), Claude-3.5-Sonnet, Gemini-1.5-Pro/2.0-Flash, DeepSeek-V3, Qwen2.5-72B, Llama-3.1-405B와 직접 비교해 short context에서는 대등하고, 대부분의 baseline이 급락하거나 아예 입력 불가능한 128k 이상 구간에서 격차를 벌린다.'
    significance: 'Linear attention이 frontier 규모에서 살아남는다는 최초의 실증으로, hybrid attention 비율을 소규모 연구 주제에서 실배포 설계점으로 끌어올리고 1M-4M token context의 경제성을 현실로 만들었다.'
    limitations: '논문 스스로 인공적인 long-context 벤치마크, 효율 상한으로 남은 1/8 softmax 성분, 코드 데이터 부족으로 인한 복잡한 coding 약세를 한계로 명시하며, (리뷰어 판단) 동일 규모 pure-softmax 모델과의 end-to-end serving 비교가 없어 실제 배포 이득은 측정이 아니라 추정에 가깝다.'
    future_work: '저자들은 더 현실적인 long-context 평가, softmax attention을 완전히 제거해 사실상 무제한 context로 가는 아키텍처 탐색, 다음 버전을 위한 코드 데이터 보강을 계획한다고 밝혔다.'
    resources: 'GitHub(MiniMax-AI)와 HuggingFace(MiniMax-Text-01/VL-01)에 weight와 코드가 공개되어 있고, Hailuo chatbot과 상용 API도 제공된다.'
  en:
    background: 'Linear attention has long promised O(n) context scaling against the quadratic compute and ever-growing KV cache of softmax attention, but it had never been validated beyond small academic models.'
    problem: 'Whether a linear-attention-dominant architecture can match state-of-the-art softmax-attention LLM quality at hundreds of billions of parameters and million-token contexts was an open, untested question.'
    prior_limits: 'Prior linear/SSM layers (TransNormer, Mamba2, HGRN2) were only trained at small scale, show clearly weak needle-retrieval (NIAH) ability, and every mature distributed training/inference stack is optimized for softmax attention alone.'
    goal: 'Build a frontier-class open model whose attention cost stays near-linear so it can train natively at 1M-token context and extrapolate to 4M tokens at inference at affordable cost.'
    method: 'MiniMax-Text-01 is a 456B-parameter / 45.9B-activated MoE transformer (80 layers, hidden 6144, 32 experts with top-2 routing) that places one GQA softmax-attention block after every seven lightning-attention blocks (an I/O-aware tiled implementation of TransNormer linear attention), applies RoPE to half the head dimensions, and is trained with LASP+, varlen ring attention, and expert-parallel overlap through a three-stage context extension to 1M tokens.'
    key_idea: 'A thin 1-in-8 dose of softmax attention restores the retrieval capability that pure linear attention lacks — the hybrid matches or surpasses full softmax attention on retrieval and length extrapolation while keeping 7/8 of the layers at linear cost.'
    validation: 'Chinchilla-style scaling-law fits over 70M-7B models trained on up to 300B tokens compare softmax, lightning, and hybrid variants, and the full model is evaluated on MMLU/MMLU-Pro/GPQA/IFEval/Arena-Hard, RULER up to 1M tokens, LongBench-v2, a 4M-token NIAH pressure test, and MTOB long in-context learning.'
    results: 'MiniMax-Text-01 reaches **88.5** MMLU and **54.4** GPQA-diamond, holds RULER **0.910** at 1M-token input, tops the evaluated set on LongBench-v2 with-CoT at **56.5**, stays near-perfect on NIAH out to **4M** tokens, and inference achieves over **75%** MFU on H20 GPUs.'
    comparison: 'It is benchmarked head-to-head against GPT-4o (11-20), Claude-3.5-Sonnet, Gemini-1.5-Pro/2.0-Flash, DeepSeek-V3, Qwen2.5-72B, and Llama-3.1-405B — roughly at parity in short context, with a widening lead beyond 128k where most baselines degrade sharply or cannot accept the input at all.'
    significance: 'It is the first demonstration that linear attention survives frontier scale, turning hybrid attention ratios from a small-model research question into a deployed design point and making 1M-4M-token context windows economically plausible.'
    limitations: 'The paper itself lists artificial long-context benchmarks, the remaining 1/8 softmax component as an efficiency bound, and weak complex-coding performance from limited code data; (reviewer judgment) it also reports no matched-scale end-to-end serving comparison against a pure-softmax peer, so the real deployment win is inferred rather than measured.'
    future_work: 'The authors state plans for more realistic long-context evaluation, architectures that eliminate softmax attention entirely toward effectively unlimited context, and stronger code-data curation for the next model version.'
    resources: 'Weights and code are public (GitHub MiniMax-AI, HuggingFace MiniMax-Text-01 and MiniMax-VL-01), alongside the Hailuo chatbot and a commercial API.'
source: 'manual'
---

## Notes

MiniMax-01 is the existence proof the hybrid-attention line needed: its 1:7 softmax-to-lightning layer ratio at 456B scale is the same underlying question DuoAttention poses at head level and Gated DeltaNet at layer level — how little full attention does retrieval actually require? The most reusable parts for efficiency work are the scaling-law comparison (hybrid reaches lower loss than pure softmax at equal compute) and the clean diagnosis that pure linear attention fails specifically at retrieval, not at language modeling. The systems contribution is equally real: LASP+ and varlen ring attention are what made 1M-token training of a linear-hybrid practical at this scale.

Hybrid 비율(1/8)과 softmax layer 배치 위치 자체에 대한 ablation은 얇은 편이라, full-vs-linear 배치 설계는 여전히 열린 연구 공간이다.
