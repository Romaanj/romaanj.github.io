---
title: 'MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases'
arxivId: '2402.14905'
authors: 'Liu et al.'
lab: 'Meta'
venue: 'ICML 2024'
date: 2026-07-04
tags: [on-device, architecture, efficiency]
topic: 'on-device'
summary: 'A systematic sub-billion design-space study showing that at 125M-350M scale deep-thin layouts, embedding sharing, GQA, and block-wise layer sharing buy accuracy per byte, yielding state-of-the-art on-device LMs.'
summary_ko: 'Sub-billion 스케일에서는 deep-thin 레이아웃, embedding sharing, GQA, block-wise layer sharing이 byte당 정확도를 산다는 것을 체계적 design-space 탐색으로 보여 on-device LM의 SoTA 레시피를 세운 논문.'
links: [kivi, minimax-01, gated-deltanet]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2402.14905' }
  - { label: 'PDF', url: 'https://arxiv.org/pdf/2402.14905' }
  - { label: 'GitHub', url: 'https://github.com/facebookresearch/MobileLLM' }
  - { label: 'HuggingFace', url: 'https://huggingface.co/facebook/MobileLLM-125M' }
analysis:
  ko:
    background: '클라우드 LLM 서빙의 비용·에너지 부담과 모바일 기기의 제약 — DRAM **6-12GB**(앱은 그 일부만 사용 가능), 에너지 약 **0.1 J/token per 1B params** — 이 sub-billion 파라미터 on-device LM의 필요를 만든다.'
    problem: '1B 미만 파라미터 예산을 어떻게 배분해야(깊이 vs 폭, embedding vs layer) 모바일 DRAM·배터리 안에서 최대 품질이 나오는지가 정의되지 않은 상태였다.'
    prior_limits: 'OPT-125M/350M, GPT-Neo, Pythia, BLOOM, Cerebras-GPT 등 기존 소형 모델은 데이터·파라미터 수가 지배한다는 scaling-law 통념에 따라 아키텍처 형태를 부차적으로 취급해 이 스케일에서 품질을 크게 놓치고 있었다.'
    goal: '추가 런타임 메모리 없이 고정된 125M/350M 파라미터 예산에서 zero-shot reasoning 정확도를 최대화하는 설계 레시피를 확립하는 것이 목표다.'
    method: '0.25T-token 탐색과 1T-token 최종 학습(32×A100, Adam)으로 SwiGLU FFN(+1.3pt), deep-thin 레이아웃(30/32 layers), input-output embedding sharing(파라미터 약 10% 절감), GQA(KV heads 3/5)를 순차 ablation하고, 인접 블록 가중치를 두 번 실행하는 block-wise immediate layer sharing으로 MobileLLM-LS를 구성한다.'
    key_idea: 'Sub-billion 스케일에서는 depth가 width를 이기고, weight를 SRAM/cache에 상주시킨 채 연산만 반복하는 layer sharing이 메모리 증가 없이 정확도를 산다 — 이 구간에서는 아키텍처 형태가 scaling-law의 무차별 예측보다 중요하다는 관측이 핵심이다.'
    validation: '8개 zero-shot commonsense 벤치마크(ARC-e/c, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande)와 TriviaQA·RACE, AlpacaEval·MT-Bench chat 평가, 합성 API-calling 태스크, iPhone 13(ExecuTorch+Metal) 지연 측정, W8A8 PTQ 호환성 체크로 검증했다.'
    results: '이전 SoTA 125M/350M 대비 평균 **+2.7%/+4.3%**, layer sharing으로 추가 **+0.7%/+0.8%**를 얻었고, API-calling exact-match intent **65.3%**로 LLaMA-v2 7B(62.8%)와 대등하며, layer sharing의 iPhone 실행 오버헤드는 **2.6%**, W8A8 PTQ 정확도 하락은 **0.5pt 미만**이다.'
    comparison: 'OPT, GPT-Neo, Pythia, RWKV, BLOOM, Cerebras-GPT, Galactica, LaMini-GPT 등 동급 전부와 더 큰 Pythia-1B·BLOOM-1.1B·Falcon-1.3B까지 이기거나 대등했고, API calling에서는 LLaMA-v2 7B급 exact-match를 보였다.'
    significance: 'Deep-thin + weight-sharing을 sub-billion on-device LM의 기본 레시피로 확립하고 흔한 on-device 태스크에 7B가 필요 없음을 보여, 이후 소형 모델 설계의 기준점이 됐다.'
    limitations: '논문 명시로는 LLaMA-v2 7B knowledge distillation이 label 학습 대비 이득 없이 학습만 **2.6-3.2×** 느렸고 API-calling Rouge는 7B보다 낮으며, (리뷰어 판단) 평가가 commonsense·chat 중심이라 long-context·다국어·코드 능력은 검증되지 않았다.'
    future_work: '논문은 명시적 후속 과제 절을 두지 않으며, (리뷰어 판단) 레시피의 상위 스케일 확장과 quantization·sparsity·KV-cache 절감 기법과의 더 깊은 결합이 자연스러운 다음 단계다.'
    resources: '학습 코드는 GitHub(facebookresearch/MobileLLM), 사전학습 체크포인트는 HuggingFace(facebook/MobileLLM-125M 등)에 공개되어 있다.'
  en:
    background: 'The cost and energy of cloud LLM serving plus hard mobile constraints — **6-12GB** DRAM of which an app may use only a fraction, and roughly **0.1 J/token per 1B params** of energy — motivate sub-billion-parameter on-device language models.'
    problem: 'It was undefined how to allocate a sub-1B parameter budget (depth vs width, embeddings vs layers) to maximize quality within mobile DRAM and battery limits.'
    prior_limits: 'Prior small models such as OPT-125M/350M, GPT-Neo, Pythia, BLOOM, and Cerebras-GPT followed the scaling-law consensus that data and parameter count dominate, treated architecture shape as secondary, and left substantial quality on the table at this scale.'
    goal: 'Establish a design recipe that maximizes zero-shot reasoning accuracy at fixed 125M/350M parameter budgets without adding runtime memory.'
    method: 'Using 0.25T-token exploration runs and 1T-token final training (32×A100, Adam), the paper sequentially ablates SwiGLU FFN (+1.3pt), deep-thin layouts (30/32 layers), input-output embedding sharing (about 10% parameter savings), and GQA (3/5 KV heads), then builds MobileLLM-LS via block-wise immediate layer sharing that executes each shared block twice.'
    key_idea: 'At sub-billion scale depth beats width, and layer sharing that repeats compute while weights stay resident in SRAM/cache buys accuracy with no memory growth — architecture shape matters here, contrary to the shape-indifference suggested by scaling laws.'
    validation: 'Validated on eight zero-shot commonsense benchmarks (ARC-e/c, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande), TriviaQA and RACE, AlpacaEval and MT-Bench chat evals, a synthetic API-calling task, iPhone 13 latency via ExecuTorch+Metal, and a W8A8 PTQ compatibility check.'
    results: 'MobileLLM gains **+2.7%/+4.3%** average accuracy over prior SoTA 125M/350M models, layer sharing adds **+0.7%/+0.8%**, API-calling exact-match intent reaches **65.3%** versus 62.8% for LLaMA-v2 7B, iPhone execution overhead from layer sharing is only **2.6%**, and W8A8 PTQ costs **under 0.5pt**.'
    comparison: 'It beats or matches OPT, GPT-Neo, Pythia, RWKV, BLOOM, Cerebras-GPT, Galactica, and LaMini-GPT at equal size, holds up against larger Pythia-1B, BLOOM-1.1B, and Falcon-1.3B, and matches LLaMA-v2 7B exact-match on API calling.'
    significance: 'It established deep-thin plus weight-sharing as the default recipe for sub-billion on-device LMs and showed common on-device tasks do not require 7B-class models, becoming a reference point for later small-model families.'
    limitations: 'The paper states that knowledge distillation from LLaMA-v2 7B gave no accuracy benefit over label training while slowing training **2.6-3.2×**, and API-calling Rouge trails 7B models; (reviewer judgment) evaluation centers on commonsense and chat, leaving long-context, multilingual, and coding ability untested.'
    future_work: 'The paper offers no explicit future-work agenda; (reviewer judgment) scaling the recipe above 350M and combining it more deeply with quantization, sparsity, and KV-cache reduction are the natural next steps.'
    resources: 'Training code is released on GitHub (facebookresearch/MobileLLM) and pretrained checkpoints on HuggingFace (facebook/MobileLLM-125M and siblings).'
source: 'manual'
---

## Notes

MobileLLM is the canonical design-space paper for sub-billion on-device LMs: instead of proposing one trick, it measures how each architectural lever (depth vs width, SwiGLU, embedding sharing, GQA, layer sharing) pays off under a hard parameter budget, and its deep-thin recipe was picked up by most later small-model families. The layer-sharing result is the most interesting lens for efficiency work — it explicitly trades extra FLOPs for weights that stay resident in cache, the same compute-for-memory-traffic exchange that governs memory-bound decode at every scale. Its GQA and W8A8 findings also connect on-device architecture design to the KV-cache and quantization line of work.

작은 모델에서 아키텍처 형태가 scaling law보다 중요해진다는 관측은, 파라미터 예산이 아니라 메모리 트래픽이 병목인 구간에서는 설계 축이 달라진다는 일반 교훈으로 읽힌다.
