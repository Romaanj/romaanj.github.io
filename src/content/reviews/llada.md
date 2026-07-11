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
figures:
  - src: '/figures/llada/fig1.png'
    caption: 'The full LLaDA recipe in three panels — (a) pre-training masks every token independently at a random ratio t~U[0,1], (b) SFT masks only response tokens, and (c) sampling walks from t=1 (fully masked) to t=0, predicting all masks and remasking the low-confidence ones each step.'
    caption_ko: '(a) pre-training은 모든 token을 t~U[0,1] 비율로 독립 masking하고, (b) SFT는 response token만 masking하며, (c) sampling은 t=1(전부 masked)에서 t=0까지 매 step 전체를 예측한 뒤 low-confidence token을 remask하는 과정을 보여준다.'
    credit: 'Figure 2 from arXiv:2502.09992 — authors'' figure'
  - src: '/figures/llada/fig2.png'
    caption: 'Benchmark radar of LLaDA 8B Base (red) against LLaMA3 8B and LLaMA2 7B — the red polygon bulges ahead on math (GSM8K, MATH) and Chinese (CMMLU, C-Eval) and dips behind on BBH and ARC-C.'
    caption_ko: 'LLaDA 8B Base(빨강) vs LLaMA3 8B·LLaMA2 7B benchmark radar — 빨간 다각형이 수학(GSM8K, MATH)·중국어(CMMLU, C-Eval)에서는 앞서고 BBH·ARC-C에서는 뒤지는 모양을 보라.'
    credit: 'Figure 1 from arXiv:2502.09992 — authors'' figure'
analysis:
  ko:
    background: >-
      Scalability, in-context learning, instruction following 같은 현대 LLM의 핵심 능력은
      사실상 autoregressive next-token prediction 레시피에서만 대규모로 검증되어 왔고, 그 결과
      AR factorization이 LLM과 동의어처럼 취급되어 왔다. 논문은 이 통념을 정면으로 겨냥해,
      scalability는 AR 고유의 산물이 아니라 Transformer·model size·data size와 generative
      principle이 유도하는 Fisher consistency의 상호작용에서 나온다고 주장한다. Masked
      diffusion 자체는 likelihood bound가 정립된 기존 접근이었으므로, 남은 것은 이 주장을
      지탱할 규모의 실증뿐이었다.
    problem: >-
      이 능력들이 left-to-right AR decoding 고유의 산물인지 generative modeling 원리 일반의
      산물인지는, non-AR 생성 LM이 경쟁 가능한 규모로 학습된 적이 없어 한 번도 실험적으로
      판별되지 않았다. 판별하려면 AR이 아닌 원리로 표준 파이프라인 전체(pretraining+SFT)를
      8B급에서 밟는 실험이 필요한데, 그 비용 때문에 질문이 공백으로 남아 있었다. 요리의 맛이
      재료를 넣는 순서에서 온다고 모두가 믿는데 정작 다른 순서로 풀 스케일 요리를 해 본 사람이
      없던 상황과 같다 — LLaDA는 그 요리를 실제로 해 본 논문이다.
    prior_limits: >-
      기존 masked diffusion LM들은 10^18–10^20 FLOPs(약 1B 이하) 규모에 머물렀고 pretrain+SFT
      전체 파이프라인 없이 학습되어, 표준 벤치마크 스위트에서 AR LLM과 정면 비교된 적이 없었다.
      선행 연구(Nie et al.)의 scaling 분석도 10^18–10^20 FLOPs 구간에서 멈췄고, 본 논문이 이를
      10^20–10^23 FLOPs로 확장했다. 게다가 likelihood는 downstream 성능의 간접 지표인 데다
      diffusion은 그 bound만 최적화하므로, ARM과 공정하게 견줄 수단은 사실상 벤치마크
      head-to-head뿐이었다.
    goal: >-
      8B 규모 masked diffusion LM을 표준 LLM 파이프라인(pretraining+SFT)으로 완전히 from
      scratch 학습하여 동급 AR baseline과 head-to-head로 비교 검증하는 것이 목표다. 특수 기법
      없이 기존 LLM과 유사한 data protocol을 따라 학습해, 성능 차이가 modeling 원리의 차이로
      귀속되도록 설계했다. 동일 데이터로 자체 학습한 compute-matched ARM baseline(1B 규모에서는
      architecture·데이터·설정까지 동일)이 scaling 비교의 기준점이 된다.
    method: >-
      Forward process가 각 token을 t~U[0,1] 비율로 독립적으로 masking하면, causal mask 없는
      Transformer mask predictor가 masked token 전부를 동시에 예측한다 — KV caching과 비호환인
      구조라 GQA 대신 vanilla MHA를 쓰고, FFN dimension을 12,288로 줄여 총 **8.02B**로 크기를
      맞췄다. 학습은 masked 위치에만 걸리는 1/t-가중 cross-entropy(negative log-likelihood의
      upper bound, Monte Carlo 추정)로 **2.3T** token pretraining(**0.13M** H800 GPU-hours)과
      prompt는 남기고 response만 masking하는 **4.5M** pair SFT를 수행한다. 추론은 전부 masked된
      response에서 시작해 예측→일부 remask를 반복하는데, 원칙상 random이어야 할 remask 대상을
      MaskGIT식으로 confidence 최하위 s/t 비율로 고르는 low-confidence remasking(선택적으로
      block 단위 semi-autoregressive)을 쓴다. 십자말풀이를 연필로 푸는 것과 같다 — 매 라운드 전
      칸을 채워 보고, 확신 있는 칸만 잉크로 굳히고, 애매한 칸은 지워 다시 푼다.
    key_idea: >-
      LLM 능력의 원천은 AR factorization이 아니라 generative modeling 원리(양방향 masked
      prediction에 대한 likelihood bound)라는 것이 핵심 주장이다. 저자들은 scalability를
      Transformer·모델 규모·데이터 규모와 Fisher consistency의 상호작용으로 설명하고, 어느
      방향으로든 예측하는 bidirectional 구조 덕에 reversal curse 같은 순서-비대칭 실패가
      구조적으로 사라진다고 본다. 부수 효과로 decoding이 iterative refinement가 되면서 sampling
      step 수가 새로운 quality–compute 조절축이 된다.
    validation: >-
      MMLU, BBH, GSM8K, MATH, HumanEval, MBPP, CMMLU, C-Eval 등 15개 표준 zero/few-shot
      벤치마크에서 동일 프로토콜로 재평가한 LLaMA3 8B/LLaMA2 7B와 비교했다. Scaling은 동일
      데이터로 자체 학습한 ARM baseline과 10^20–10^23 FLOPs 구간, 6개 task(MMLU, GSM8K, PIQA,
      MATH, HumanEval, MBPP)에서 pre-training compute를 단일 scaling 지표로 놓고 검증했다.
      여기에 한시(poem) forward/reversal completion으로 GPT-4o·Qwen2.5 대비 reversal curse를
      별도로 검증했다.
    results: >-
      LLaDA 8B Base는 MMLU **65.9**(LLaMA3 8B 65.4), GSM8K **70.3**(vs **48.7**), MATH
      **31.4**(vs 16.0), HumanEval **35.4**(vs 34.8)를 기록했고, 중국어 CMMLU **69.9**(vs
      50.7)·C-Eval **70.5**(vs 51.7)에서는 크게 앞섰다. Instruct 모델은 reversal poem
      completion에서 GPT-4o를 **45.6** vs **34.3**으로 앞섰는데, 승패보다 대칭성이 요점이다 —
      GPT-4o는 forward 82.7에서 reversal 34.3으로 붕괴하는 반면 LLaDA는 51.8→45.6으로 방향에
      거의 무관하다. Scaling 실험에서도 10^23 FLOPs까지 ARM baseline과 대등한 전반적 추세를
      보였다.
    comparison: >-
      LLaMA3 8B, LLaMA2 7B(둘 다 자체 재평가), Qwen2/2.5 7B, Mistral 7B, Deepseek 7B와 비교해
      BBH(**49.7** vs 62.1)·ARC-C(45.9 vs 53.1) 등 일부에서는 뒤지지만, 학습 token이 LLaMA3의
      15T(Qwen2.5는 18T) 대비 **2.3T**뿐인데도 수학·중국어 task에서는 앞선다. Instruct 비교는
      더 불리한 조건인데, LLaDA는 SFT만 거친 반면 비교 대상은 모두 SFT+RL alignment를 거쳤고,
      저자들도 RL 미적용 탓에 LLaMA3 8B Instruct에 다소 뒤진다고 명시한다. 표에 훈련 token 수와
      post-training 방식을 병기해 이 비대칭을 투명하게 드러낸 점은 평가할 만하다(리뷰어 판단).
    significance: >-
      Non-autoregressive diffusion 학습 LM이 8B 규모에서 경쟁력 있다는 최초의 존재 증명으로서
      parallel decoding·any-order generation을 다루는 dLLM 연구 계열을 정당화했다. 4분 마일
      기록처럼 수치 자체보다 "가능하다"는 사실이 판을 바꾼 경우다 — 이후 dLLM 연구는 전부 이 존재
      증명 위에 서 있다. Fine-tuning 없이 reversal curse가 사라진 것은 factorization 선택이
      능력의 형태까지 결정한다는 드문 인과적 증거이고, 동시에 bidirectional attention이 exact KV
      cache를 깨기 때문에 새로운 inference-efficiency 문제 축이 열렸다.
    limitations: >-
      논문 스스로 추론에 KV cache류 system-level 최적화가 없어 연산 비용이 크고, generation
      length가 사용자 지정 hyperparameter이며, RL alignment 미적용, diffusion 특화
      attention/position embedding 부재, ARM baseline과의 scaling 비교가 10^23 FLOPs 이하로
      제한됨을 명시한다. 실질적 병목은 vanilla decoding이 매 step 전체 sequence에 대한
      bidirectional attention을 재계산한다는 점이다 — 한 줄 고칠 때마다 문서 전체를 처음부터
      다시 읽는 셈이라 wall-clock에서 KV-cached AR에 크게 불리하다(리뷰어 판단). SFT 후 MMLU 등
      일부 지표가 오히려 하락한 점도 SFT 데이터 품질 문제로 추정만 하고 넘어간다.
    future_work: >-
      논문은 추가 scaling, RL 기반 alignment, multimodal 확장, agent 시스템 통합, O1류
      post-training, 그리고 더 효율적이고 제어 가능한 sampling 알고리즘을 향후 과제로 제시한다.
      이 중 추론 가속(caching·parallel decoding)이 가장 빠르게 채워진 전선으로, Fast-dLLM류
      후속 연구의 직접적 출발점이 되었다(리뷰어 판단).
    resources: >-
      GitHub에 코드와 project page가, HuggingFace에 LLaDA-8B Base/Instruct checkpoint가
      공개되어 있다. 논문 Appendix에 pretraining·SFT·sampling·conditional likelihood 평가의
      pseudo-code 알고리즘이 수록되어 있어 재현 경로가 명확하다.
  en:
    background: >-
      Scalability, in-context learning, and instruction following — the capabilities that
      define modern LLMs — had been demonstrated at scale almost exclusively under the
      autoregressive next-token-prediction recipe, making AR factorization look synonymous
      with the term LLM. The paper targets this belief head-on, arguing that scalability comes
      from the interplay of Transformers, model size, data size, and the Fisher consistency
      induced by the generative principle, rather than from AR modeling per se. Masked
      diffusion itself was an established approach with a well-defined likelihood bound; what
      was missing was evidence at scale.
    problem: >-
      Whether these capabilities are intrinsic to left-to-right AR decoding or follow from
      generative-modeling principles in general had never been discriminated experimentally,
      because no non-AR generative LM had been trained at competitive scale. The
      discriminating experiment requires running the full standard pipeline (pretraining +
      SFT) at 8B scale under a non-AR principle, and that cost kept the question open. It is
      as if everyone credited a dish''s flavor to the order the ingredients go in, while
      nobody had ever cooked it in a different order at full scale — LLaDA is the paper that
      actually cooks it.
    prior_limits: >-
      Prior masked-diffusion LMs stayed in the 10^18–10^20 FLOPs regime (roughly 1B parameters
      or below) without a full pretrain-plus-SFT pipeline, so they never faced AR LLMs on
      standard benchmark suites. The earlier scaling analysis of Nie et al. also stopped at
      10^18–10^20 FLOPs, which this paper extends to 10^20–10^23 FLOPs. Likelihood, moreover,
      is an indirect proxy for downstream ability, and diffusion optimizes only a bound on it,
      so benchmark head-to-head was effectively the only fair comparison left.
    goal: >-
      Train a masked-diffusion language model entirely from scratch at the 8B scale through
      the standard LLM pipeline (pretraining + SFT) and test head-to-head whether it matches
      same-scale AR baselines. No special techniques are used and the data protocol follows
      existing LLM practice, so that any capability gap is attributable to the modeling
      principle itself. Compute-matched ARM baselines trained by the authors on the same data
      (identical architecture, data, and configuration at the 1B scale) anchor the scaling
      comparison.
    method: >-
      A forward process masks each token independently at ratio t~U[0,1], and a
      causal-mask-free Transformer mask predictor predicts all masked tokens simultaneously —
      vanilla MHA replaces GQA (the architecture is incompatible with KV caching anyway), with
      the FFN dimension trimmed to 12,288 to keep the total at **8.02B** parameters. Training
      minimizes a 1/t-weighted cross-entropy over masked positions only — an upper bound on
      negative log-likelihood, estimated via Monte Carlo — through **2.3T**-token pretraining
      (**0.13M** H800 GPU-hours) plus **4.5M**-pair SFT in which prompts stay visible and only
      response tokens are masked. Decoding starts from a fully masked response and iterates
      predict-then-remask; the remasking should in principle be random, but a MaskGIT-style
      low-confidence rule instead remasks the s/t fraction of least-confident predictions
      (optionally block-wise, i.e., semi-autoregressive). It works like solving a crossword in
      pencil: fill every square each round, ink in the answers you trust, and erase the shaky
      ones to retry.
    key_idea: >-
      The claim is that the generative-modeling principle — a likelihood bound over
      bidirectional masked prediction — rather than AR factorization itself delivers LLM
      capabilities. The authors ground scalability in the interplay of Transformers, model
      scale, data scale, and Fisher consistency, and the bidirectional any-order structure
      removes order-asymmetric failures like the reversal curse by construction. As a side
      effect, decoding becomes iterative refinement, making the sampling-step count a new
      quality–compute knob.
    validation: >-
      Evaluated on 15 standard zero/few-shot benchmarks (MMLU, BBH, GSM8K, MATH, HumanEval,
      MBPP, CMMLU, C-Eval, etc.) against LLaMA3 8B and LLaMA2 7B re-evaluated under the same
      protocol. Scaling is tested against ARM baselines trained by the authors on the same
      data over 10^20–10^23 FLOPs on six tasks (MMLU, GSM8K, PIQA, MATH, HumanEval, MBPP),
      with pre-training compute as the single scaling metric. A Chinese poem forward/reversal
      completion probe against GPT-4o and Qwen2.5 separately targets the reversal curse.
    results: >-
      LLaDA 8B Base scores MMLU **65.9** (LLaMA3 8B: 65.4), GSM8K **70.3** (vs **48.7**), MATH
      **31.4** (vs 16.0), and HumanEval **35.4** (vs 34.8), with large leads on Chinese suites
      — CMMLU **69.9** (vs 50.7) and C-Eval **70.5** (vs 51.7). The Instruct model beats
      GPT-4o on reversal poem completion at **45.6** vs **34.3**, and the symmetry matters
      more than the win: GPT-4o collapses from 82.7 forward to 34.3 reversal while LLaDA stays
      nearly direction-invariant at 51.8→45.6. In the scaling study LLaDA tracks the ARM
      baselines up to 10^23 FLOPs with a highly competitive overall trend.
    comparison: >-
      Against LLaMA3 8B, LLaMA2 7B (both re-evaluated in-house), Qwen2/2.5 7B, Mistral 7B, and
      Deepseek 7B it trails on suites like BBH (**49.7** vs 62.1) and ARC-C (45.9 vs 53.1),
      yet leads on math and Chinese tasks despite training on only **2.3T** tokens versus 15T
      for LLaMA3 (18T for Qwen2.5). The Instruct comparison is even more lopsided: LLaDA had
      SFT only while every comparator adds RL alignment, and the authors state their results
      trail LLaMA3 8B Instruct partly for this reason. The tables disclose training tokens and
      post-training regimes side by side, which makes the asymmetry transparent (reviewer
      judgment).
    significance: >-
      It is the first existence proof that a non-autoregressive, diffusion-trained LM is
      competitive at 8B scale, legitimizing the dLLM line of work (parallel decoding,
      any-order generation). Like the four-minute mile, the point is less the number than the
      proof of possibility — subsequent dLLM research all stands on this result. The
      disappearance of the reversal curse without any fine-tuning is rare causal evidence that
      the factorization choice shapes what a model can do, and since bidirectional attention
      breaks exact KV caching, the paper simultaneously opened a new inference-efficiency
      axis.
    limitations: >-
      The paper states that inference lacks KV-cache-style system-level optimization and is
      compute-heavy, generation length is a user-set hyperparameter, no RL alignment was
      applied, no diffusion-specialized attention or position embeddings were designed, and
      the ARM-baseline scaling comparison stops below 10^23 FLOPs. The practical bottleneck is
      that vanilla decoding recomputes full bidirectional attention over the sequence at every
      step — like re-reading an entire document from the top after each single edit — leaving
      wall-clock far behind KV-cached AR decoding (reviewer judgment). The paper also notes
      that a few metrics such as MMLU declined after SFT, attributed only speculatively to SFT
      data quality.
    future_work: >-
      The paper names further scaling, RL-based alignment, multimodal extension, agent-system
      integration, O1-style post-training, and more efficient, controllable sampling
      algorithms as next steps. Of these, inference acceleration (caching, parallel decoding)
      was the frontier that filled fastest, becoming the direct starting point for
      Fast-dLLM-style follow-up work (reviewer judgment).
    resources: >-
      Code and the project page are released on GitHub, and LLaDA-8B Base/Instruct checkpoints
      are available on HuggingFace. The paper appendix includes pseudo-code algorithms for
      pretraining, SFT, sampling, and conditional likelihood evaluation, making the
      reproduction path explicit.
source: 'manual'
---

## Notes

LLaDA is the reference starting point of the diffusion-LLM literature: most subsequent dLLM work — parallel and block-wise decoding, approximate caching, step reduction — is defined relative to its masked-diffusion formulation and its inference cost profile. Its most consequential engineering property is what it removes: bidirectional attention admits no exact KV cache, so every demasking step re-attends over the whole sequence, which is precisely the gap that Fast-dLLM-style acceleration targets. It is also a clean evidence point in the AR-versus-diffusion debate, since capabilities tracked the likelihood-bound objective plus data and compute scale rather than the factorization order. dLLM 가속 연구의 사실상 공통 baseline이며, sampling step 수라는 quality–compute 축이 여기서 처음 실전 규모로 등장했다.
