---
title: "Nemotron-Labs-Diffusion: A Tri-Mode LM Unifying Autoregressive, Diffusion, and Self-Speculation Decoding"
arxivId: "2607.05722"
authors: "Yonggan Fu, Lexington Whalen, Abhinav Garg, Chengyue Wu, ... Song Han, Pavlo Molchanov, Jan Kautz (26 authors)"
lab: "NVIDIA"
date: 2026-07-08
tags: ["dllm", "architecture", "serving", "hybrid-decoding"]
topic: 'diffusion-llm'
summary: "A single NVIDIA model switches between autoregressive, block-wise diffusion, and self-speculative decoding from one set of weights, with self-speculation (diffusion drafts, the same model's AR pass verifies) beating multi-token prediction and the 8B model decoding 6x more tokens per forward than Qwen3-8B at comparable accuracy."
summary_ko: 'NVIDIA의 한 모델이 하나의 가중치로 autoregressive, block-wise diffusion, self-speculative 디코딩 세 모드를 전환하며, self-speculation(diffusion이 초안을 만들고 같은 모델의 AR 패스가 검증)이 multi-token prediction보다 우수하고 8B 모델은 비슷한 정확도에서 Qwen3-8B 대비 forward당 6배 많은 토큰을 디코딩한다.'
links: ["fast-dllm", "llada", "set-diffusion", "sangam"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.05722' }
  - { label: 'Code (GitHub)', url: 'https://github.com/NVlabs/Nemotron-Labs-Diffusion' }
  - { label: 'Checkpoints (Hugging Face)', url: 'https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B' }
analysis:
  ko:
    background: 'Autoregressive(AR) LM은 좌→우 순차 생성이라 스텝당 토큰이 하나뿐이고, diffusion 방식 LM은 여러 위치를 동시에 정제해 병렬성을 얻지만 두 방식은 보통 서로 다른 모델로 별도 학습된다.'
    problem: '단일 디코딩 모드로 고정된 모델은 배포 환경(동시성, 지연 요구)에 따라 최적 모드가 달라져도 전환할 수 없고, multi-token prediction(MTP) 같은 speculative 기법은 별도의 보조 head가 필요하며 수용률(acceptance rate) 면에서 한계가 있다.'
    prior_limits: '기존 연구는 AR 전용 모델 또는 diffusion 전용 모델을 각각 학습하거나, self-speculation을 별도의 draft 모델·보조 MTP head로 구현했다(리뷰어 판단: 한 모델이 AR·diffusion·self-speculation 세 모드를 모두 지원한 사례는 이 논문 이전에는 명시적으로 비교되지 않는다).'
    goal: '하나의 가중치 집합으로 AR 디코딩, block-wise diffusion 디코딩, self-speculation(같은 모델이 초안과 검증을 모두 수행)을 모두 지원해 배포 상황에 따라 모드를 전환하며 높은 처리량을 유지하는 것.'
    method: 'Ministral3-8B를 기반으로 2단계 continued pretraining(1단계 순수 AR, 2단계 AR+diffusion 결합 objective, `L = L_AR + α·L_diff`, α=0.3)을 수행하고, noised 스트림과 clean 스트림을 함께 처리하되 clean→clean 어텐션을 엄격히 causal로 유지하는 **dual-stream attention** 구조로 한 번의 forward-backward pass에서 두 objective를 label 누출 없이 동시에 계산한다.'
    key_idea: 'clean 스트림을 엄격히 causal로 유지하면 AR objective와 diffusion objective를 같은 forward-backward pass에서 label 누출 없이 함께 학습할 수 있어, diffusion 학습이 주는 lookahead/planning 능력과 AR 학습이 주는 좌→우 선형 사전지식이 한 모델 안에서 상호 보완적으로 공존한다.'
    validation: '256장의 H100에서 3B/8B/14B 규모(base/instruct/VLM)로 학습해 Qwen3-8B(AR)·SDAR-8B(diffusion) 등 오픈소스 베이스라인과 비교하고, "recursive dynamic compaction"이라는 oracle 기반 절차로 모델의 이론적 병렬성 한계(Speed-of-Light, SOL)를 별도로 측정했으며, GB200 GPU에서 SGLang을 통한 SPEED-Bench로 실제 처리량을 측정했다.'
    results: 'Nemotron-Labs-Diffusion-8B는 Qwen3-8B 대비 비슷한 정확도에서 forward당 **6배** 많은 토큰을 디코딩해 SPEED-Bench에서 **4배** 높은 처리량을 내며, self-speculation의 quadratic 변형은 forward당 최대 **6.38배**의 토큰 생성을 달성하고, SOL 분석은 최적 sampler를 가정하면 현재 self-speculation 대비 forward당 최대 **76.5%** 더 많은 토큰이 이론적으로 가능함을 보인다.'
    comparison: '논문은 Qwen3-8B(AR)·SDAR-8B(diffusion)와 정확도·속도를 직접 비교하고, self-speculation을 MTP 기반 speculative decoding과 비교해 수용률과 실기기 효율 양쪽에서 MTP를 능가한다고 보고한다.'
    significance: '하나의 체크포인트가 AR·diffusion·self-speculation을 모두 지원함을 보여 배포 시나리오별로 별도 모델을 학습·서빙할 필요를 없앨 수 있음을 시사하며, SOL 분석은 diffusion 모델이 "얼마나 병렬화될 수 있는가"와 "현재 sampler가 실제로 달성하는 병렬성"을 분리해 측정한 사례로 유용하다.'
    limitations: 'SOL 분석이 보여주는 76.5%의 잠재적 개선폭은 아직 실현되지 않은 상한선이며, 이는 곧 현재 sampler가 diffusion 모드의 이론적 병렬성을 완전히 실현하지 못하고 있음을 논문 스스로 인정하는 대목이다(리뷰어 판단: 어떤 sampler 설계 변화가 이 격차를 좁힐지는 abstract 수준에서는 명시되지 않는다).'
    future_work: '논문은 SOL 상한에 근접하는 것을 앞으로의 과제로 제시한다(리뷰어 판단: quadratic self-speculation처럼 이론적 처리량이 더 높은 변형의 커널 최적화, 그리고 다른 하드웨어/배치 조건에서의 검증도 자연스러운 다음 단계로 보인다).'
    resources: '코드는 GitHub `NVlabs/Nemotron-Labs-Diffusion`에 공개되어 있고, 3B/8B/14B(base/instruct/VLM) 체크포인트가 Hugging Face `nvidia/Nemotron-Labs-Diffusion-*`에 공개되어 있다.'
  en:
    background: 'Autoregressive (AR) LMs generate strictly left-to-right, producing one token per step, while diffusion-style LMs refine multiple positions in parallel — but the two are normally trained as separate models.'
    problem: 'A model fixed to one decoding mode cannot switch to the mode best suited to a given deployment setting (concurrency, latency budget), and speculative techniques such as multi-token prediction (MTP) require dedicated auxiliary heads with limited acceptance rates.'
    prior_limits: 'Prior work trains dedicated AR-only or diffusion-only models, or implements self-speculation with a separate draft model or auxiliary MTP heads (reviewer judgment: a single model supporting AR, diffusion, and self-speculation together is not shown as directly compared before this paper).'
    goal: 'Support AR decoding, block-wise diffusion decoding, and self-speculation (the same model drafts and verifies) from one set of weights, switching modes to sustain high throughput across deployment settings.'
    method: 'Starting from a Ministral3-8B base, two-stage continued pretraining (stage 1 pure AR, stage 2 a joint objective `L = L_AR + α·L_diff`, α=0.3) is combined with a **dual-stream attention** design that processes a noised stream and a clean stream together while keeping clean-to-clean attention strictly causal, letting both objectives be computed in a single forward-backward pass without label leakage.'
    key_idea: 'Keeping the clean stream strictly causal lets the AR and diffusion objectives be trained jointly in one pass without leaking future labels, so the lookahead/planning ability diffusion training provides and the left-to-right prior AR training provides end up complementary within a single model rather than competing.'
    validation: 'Trained on 256 H100s at 3B/8B/14B scale (base/instruct/VLM variants), compared against open-source baselines including Qwen3-8B (AR) and SDAR-8B (diffusion), with a separate oracle-based procedure ("recursive dynamic compaction") measuring the theoretical parallelism ceiling of the model (speed-of-light, SOL), and real throughput measured via SGLang on a GB200 GPU using SPEED-Bench.'
    results: 'Nemotron-Labs-Diffusion-8B decodes **6x** more tokens per forward than Qwen3-8B at comparable accuracy, translating to **4x** higher SPEED-Bench throughput; the quadratic self-speculation variant reaches up to **6.38x** tokens per forward; the SOL analysis shows up to **76.5%** more tokens per forward are theoretically available versus current self-speculation under an optimal sampler.'
    comparison: 'The paper directly compares accuracy and speed against Qwen3-8B (AR) and SDAR-8B (diffusion), and compares self-speculation against MTP-based speculative decoding, reporting it exceeds MTP in both acceptance rate and real-device efficiency.'
    significance: 'It demonstrates that one checkpoint can support AR, diffusion, and self-speculation, potentially removing the need to train and serve separate models per deployment scenario, and the SOL analysis usefully separates "how parallel the model could be" from "how much parallelism the current sampler actually realizes."'
    limitations: 'The 76.5% headroom shown by the SOL analysis is an unrealized ceiling — the paper itself frames this as evidence that the current sampler does not fully realize the theoretical parallelism of diffusion mode (reviewer judgment: which specific sampler design changes would close this gap is not specified at the abstract level).'
    future_work: 'The paper names approaching the SOL ceiling as future work (reviewer judgment: kernel optimization for higher-theoretical-throughput variants like quadratic self-speculation, and validation under other hardware/batching conditions, are natural next steps).'
    resources: 'Code is public on GitHub at `NVlabs/Nemotron-Labs-Diffusion`, and 3B/8B/14B (base/instruct/VLM) checkpoints are public on Hugging Face under `nvidia/Nemotron-Labs-Diffusion-*`.'
source: "autosweep"
---

## Notes

The tech report behind this paper first circulated in May 2026; this arXiv posting (plus the public
GitHub code and Hugging Face checkpoints) is what makes it independently verifiable and citable.
The most transferable idea outside NVIDIA's own model is the dual-stream trick for getting an AR
objective and a diffusion objective into the same forward-backward pass without one leaking labels
into the other — that constraint (keep the "clean" stream strictly causal) is a reusable recipe for
anyone trying to co-train AR and diffusion decoding on one backbone, independent of the rest of this
paper's architecture choices.
