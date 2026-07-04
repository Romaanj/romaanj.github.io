---
title: 'LLaDA: Large Language Diffusion Models'
arxivId: '2502.09992'
authors: 'Nie et al.'
lab: 'RUC GSAI / Ant Group'
venue: 'ICML 2025'
date: 2026-06-28
tags: [diffusion-llm, architecture]
topic: diffusion-llm
summary: 'An 8B masked-diffusion language model trained from scratch on 2.3T tokens that matches LLaMA3-8B-class autoregressive baselines, showing next-token prediction is not the only recipe for LLM capabilities.'
summary_ko: '2.3T token으로 from scratch 학습한 8B masked diffusion LM이 LLaMA3 8B급 autoregressive baseline과 대등한 성능을 내며, LLM 능력이 next-token prediction만의 산물이 아님을 보인 논문.'
links: [fast-dllm]
resources:
  - label: arXiv
    url: 'https://arxiv.org/abs/2502.09992'
  - label: GitHub
    url: 'https://github.com/ML-GSAI/LLaDA'
  - label: Project
    url: 'https://ml-gsai.github.io/LLaDA-demo/'
  - label: HuggingFace
    url: 'https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct'
analysis:
  ko:
    background: 'Scalability, in-context learning, instruction following 같은 현대 LLM의 핵심 능력은 사실상 autoregressive next-token prediction 레시피에서만 대규모로 검증되어 왔고, 그 결과 AR factorization이 LLM과 동의어처럼 취급되어 왔다.'
    problem: '이 능력들이 left-to-right AR decoding 고유의 산물인지 generative modeling 원리 일반의 산물인지는, non-AR 생성 LM이 경쟁 가능한 규모로 학습된 적이 없어 한 번도 실험적으로 판별되지 않았다.'
    prior_limits: '기존 masked diffusion LM들은 10^18–10^20 FLOPs(약 1B 이하) 규모에 머물렀고 pretrain+SFT 전체 파이프라인 없이 학습되어, 표준 벤치마크 스위트에서 AR LLM과 정면 비교된 적이 없었다.'
    goal: '8B 규모 masked diffusion LM을 표준 LLM 파이프라인(pretraining+SFT)으로 완전히 from scratch 학습하여 동급 AR baseline과 head-to-head로 비교 검증하는 것이 목표다.'
    method: 'Forward process가 각 token을 t~U[0,1] 비율로 독립적으로 masking하면 causal mask 없는 Transformer(GQA 대신 vanilla MHA, 나머지는 LLaMA류)가 masked token 전부를 동시에 예측하고, negative log-likelihood의 upper bound인 1/t 가중 cross-entropy로 **2.3T** token pretraining(**0.13M** H800 GPU-hours)과 **4.5M** pair SFT를 수행하며, 추론은 전부 masked된 response에서 시작해 low-confidence remasking(선택적으로 block 단위 semi-autoregressive)으로 반복 demasking한다.'
    key_idea: 'LLM 능력의 원천은 AR factorization이 아니라 generative modeling 원리(양방향 masked prediction에 대한 likelihood bound)라는 것이 핵심 주장이며, decoding이 iterative refinement가 되면서 sampling step 수가 새로운 quality–compute 조절축이 된다.'
    validation: 'MMLU, BBH, GSM8K, MATH, HumanEval, MBPP, CMMLU, C-Eval 등 15개 표준 zero/few-shot 벤치마크에서 동일 프로토콜로 재평가한 LLaMA3 8B/LLaMA2 7B와 비교하고, 10^23 FLOPs까지 compute-matched AR baseline 대비 scaling 실험과 GPT-4o 대비 한시(poem) reversal completion 실험으로 검증했다.'
    results: 'LLaDA 8B Base는 MMLU **65.9**(LLaMA3 8B 65.4), GSM8K **70.3**(vs **48.7**), MATH **31.4**(vs 16.0), HumanEval **35.4**(vs 34.8)를 기록했고, Instruct 모델은 reversal poem completion에서 GPT-4o를 **45.6** vs **34.3**으로 앞섰다.'
    comparison: 'LLaMA3 8B, LLaMA2 7B(둘 다 자체 재평가), Qwen2/2.5 7B, Mistral 7B, Deepseek 7B와 비교해 BBH(49.7 vs 62.1)·ARC-C 등 일부에서는 뒤지지만, 학습 token이 LLaMA3의 15T 대비 **2.3T**뿐인데도 수학·중국어 task에서는 앞선다.'
    significance: 'Non-autoregressive diffusion 학습 LM이 8B 규모에서 경쟁력 있다는 최초의 존재 증명으로서 parallel decoding·any-order generation을 다루는 dLLM 연구 계열을 정당화했고, bidirectional attention이 exact KV cache를 깨기 때문에 새로운 inference-efficiency 문제 축을 열었다.'
    limitations: '논문 스스로 추론에 KV cache류 system-level 최적화가 없어 연산 비용이 크고, generation length가 사용자 지정 hyperparameter이며, RL alignment 미적용, diffusion 특화 attention/position embedding 부재, 비교가 10^23 FLOPs 이하로 제한됨을 명시하며, vanilla decoding이 매 step 전체 sequence에 대한 bidirectional attention을 재계산해 wall-clock에서 KV-cached AR에 크게 불리한 점이 실질적 병목이다(리뷰어 판단).'
    future_work: '논문은 추가 scaling, RL 기반 alignment, multimodal 확장, agent 시스템 통합, O1류 post-training을 향후 과제로 제시하며, caching·parallel decoding 등 추론 가속이 Fast-dLLM류 후속 연구의 직접적 전선이 되었다(리뷰어 판단).'
    resources: 'GitHub에 코드와 project page가, HuggingFace에 LLaDA-8B Base/Instruct checkpoint가 공개되어 있다.'
  en:
    background: 'Scalability, in-context learning, and instruction following — the capabilities that define modern LLMs — had been demonstrated at scale almost exclusively under the autoregressive next-token-prediction recipe, making AR factorization look synonymous with the term LLM.'
    problem: 'Whether these capabilities are intrinsic to left-to-right AR decoding or follow from generative-modeling principles in general had never been discriminated experimentally, because no non-AR generative LM had been trained at competitive scale.'
    prior_limits: 'Prior masked-diffusion LMs stayed in the 10^18–10^20 FLOPs regime (roughly 1B parameters or below) without a full pretrain-plus-SFT pipeline, so they never faced AR LLMs on standard benchmark suites.'
    goal: 'Train a masked-diffusion language model entirely from scratch at the 8B scale through the standard LLM pipeline (pretraining + SFT) and test head-to-head whether it matches same-scale AR baselines.'
    method: 'A forward process masks each token independently at ratio t~U[0,1], a causal-mask-free Transformer (vanilla MHA instead of GQA, otherwise LLaMA-style) predicts all masked tokens simultaneously under a 1/t-weighted cross-entropy that upper-bounds negative log-likelihood, trained with **2.3T**-token pretraining (**0.13M** H800 GPU-hours) plus **4.5M**-pair SFT, and decoded by iterative demasking from a fully masked response using low-confidence remasking (optionally block-wise semi-autoregressive).'
    key_idea: 'The claim is that the generative-modeling principle — a likelihood bound over bidirectional masked prediction — rather than AR factorization itself delivers LLM capabilities, and recasting decoding as iterative refinement makes the sampling-step count a new quality–compute knob.'
    validation: 'Evaluated on 15 standard zero/few-shot benchmarks (MMLU, BBH, GSM8K, MATH, HumanEval, MBPP, CMMLU, C-Eval, etc.) against LLaMA3 8B and LLaMA2 7B re-evaluated under the same protocol, plus a compute-matched scaling study against AR baselines up to 10^23 FLOPs and a Chinese poem reversal-completion probe against GPT-4o.'
    results: 'LLaDA 8B Base scores MMLU **65.9** (LLaMA3 8B: 65.4), GSM8K **70.3** (vs **48.7**), MATH **31.4** (vs 16.0), and HumanEval **35.4** (vs 34.8), while the Instruct model beats GPT-4o on reversal poem completion at **45.6** vs **34.3**.'
    comparison: 'Against LLaMA3 8B, LLaMA2 7B (both re-evaluated in-house), Qwen2/2.5 7B, Mistral 7B, and Deepseek 7B it trails on suites like BBH (49.7 vs 62.1) and ARC-C, yet leads on math and Chinese tasks despite training on only **2.3T** tokens versus 15T for LLaMA3.'
    significance: 'It is the first existence proof that a non-autoregressive, diffusion-trained LM is competitive at 8B scale, legitimizing the dLLM line of work (parallel decoding, any-order generation) and opening a new inference-efficiency axis since bidirectional attention breaks exact KV caching.'
    limitations: 'The paper states that inference lacks KV-cache-style system-level optimization and is compute-heavy, generation length is a user-set hyperparameter, no RL alignment was applied, no diffusion-specialized attention or position embeddings were designed, and comparisons stop below 10^23 FLOPs; in practice vanilla decoding recomputes full bidirectional attention over the sequence every step, leaving wall-clock far behind KV-cached AR decoding (reviewer judgment).'
    future_work: 'The paper names further scaling, RL-based alignment, multimodal extension, agent-system integration, and O1-style post-training as next steps, and inference acceleration via caching and parallel decoding became the immediate follow-up frontier for Fast-dLLM-style work (reviewer judgment).'
    resources: 'Code and the project page are released on GitHub, and LLaDA-8B Base/Instruct checkpoints are available on HuggingFace.'
source: 'manual'
---

## Notes

LLaDA is the reference starting point of the diffusion-LLM literature: most subsequent dLLM work — parallel and block-wise decoding, approximate caching, step reduction — is defined relative to its masked-diffusion formulation and its inference cost profile. Its most consequential engineering property is what it removes: bidirectional attention admits no exact KV cache, so every demasking step re-attends over the whole sequence, which is precisely the gap that Fast-dLLM-style acceleration targets. It is also a clean evidence point in the AR-versus-diffusion debate, since capabilities tracked the likelihood-bound objective plus data and compute scale rather than the factorization order. dLLM 가속 연구의 사실상 공통 baseline이며, sampling step 수라는 quality–compute 축이 여기서 처음 실전 규모로 등장했다.
