---
title: "FourTune: Towards Fully 4-Bit Efficient Post-Training for Diffusion Models"
arxivId: "2607.05711"
authors: "Bowen Xue, Zihan Min, Xingyang Li, Zhekai Zhang, Haocheng Xi, Lvmin Zhang, Maneesh Agrawala, Jun-Yan Zhu, Song Han, Yujun Lin, Muyang Li"
venue: "ICML 2026"
date: 2026-07-10
tags: ["quantization", "diffusion-models", "post-training", "lora"]
topic: 'compression'
summary: "An end-to-end W4A4G4 post-training framework for large diffusion models augments LoRA with a frozen numerical stabilizer to isolate quantization-sensitive outliers, matching full-precision fine-tuning quality on FLUX.1-dev (12B) with 2.25x less memory and 2.27x higher training throughput."
summary_ko: "대형 diffusion 모델을 위한 end-to-end W4A4G4 post-training 프레임워크로, LoRA에 양자화에 민감한 outlier를 격리하는 고정된(frozen) 수치 안정화 장치를 더해, FLUX.1-dev(12B)에서 full-precision 파인튜닝과 동등한 품질을 2.25배 적은 메모리와 2.27배 높은 학습 처리량으로 달성한다."
links: ["nemotron-puzzle"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.05711' }
  - { label: 'ICML 2026 poster', url: 'https://icml.cc/virtual/2026/poster/64004' }
analysis:
  ko:
    background: 'Diffusion 모델은 고품질 생성 모델링의 주류 패러다임이 되었고, 이를 다양한 다운스트림 응용에 맞추려면 post-training(파인튜닝)이 필수적이다.'
    problem: '대형 diffusion 모델의 post-training은 메모리 사용량이 과도하고 학습 속도가 느려 여전히 어려운데, 기존의 parameter-efficient 파인튜닝 기법들은 이 문제를 부분적으로만 해결한다.'
    prior_limits: '논문 내 명시 없음 — abstract 수준에서는 기존 parameter-efficient 기법(예: 표준 LoRA)이 정확히 어느 부분에서 메모리/속도 문제를 해결하지 못하는지 구체적으로 서술하지 않는다.'
    goal: '표준 LoRA 파인튜닝과 동일한 품질을 유지하면서, native 4-bit 연산으로 post-training 자체의 메모리와 속도를 획기적으로 개선하는 것.'
    method: '**FourTune**은 end-to-end **W4A4G4**(가중치·활성값·그래디언트 모두 4비트) 패러다임에 기반한 post-training 프레임워크다. 표준 LoRA 구조에 **고정된(frozen) 수치 안정화 장치(numerical stabilizer)**를 더한 3-분기(triple-branch) hybrid 파이프라인으로 양자화에 민감한 outlier를 분리해, native 4-bit 연산 하에서도 학습이 안정적으로 이루어지게 한다. 여기에 하드웨어 효율적인 block-wise quantization과 맞춤형 fused kernel을 더해 4비트 역전파(backpropagation)를 지원하고 메모리 대역폭 오버헤드를 줄인다.'
    key_idea: '양자화에 민감한 outlier를 별도의 고정 분기로 격리해 학습 가능한 LoRA 경로는 그대로 4비트로 두면서도, 이 outlier 분기가 4비트 학습을 수치적으로 불안정하게 만드는 요인을 흡수하게 하는 것이 핵심이다.'
    validation: 'Customization, reinforcement learning, distillation 세 종류의 downstream task에서 평가했으며, 12B 파라미터의 FLUX.1-dev에 적용해 BF16 LoRA와 비교했다.'
    results: '세 task 전반에서 full-precision 파인튜닝과 동등한 품질을 유지하면서, FLUX.1-dev(12B) 기준 BF16 LoRA 대비 메모리 오버헤드를 **2.25배** 줄이고 end-to-end 학습 처리량을 **2.27배** 높였다.'
    comparison: '주된 비교 대상은 BF16(FP16 계열) LoRA 파인튜닝이며, abstract 수준에서는 다른 4-bit diffusion 양자화 기법(예: 추론 전용 PTQ 기법)과의 직접 비교 수치는 제시되지 않는다.'
    significance: '추론 단계 양자화가 아니라 **post-training(학습) 자체**를 4비트로 만들어 대형 diffusion 모델 파인튜닝의 메모리·속도 장벽을 낮췄다는 점에서, on-device/저자원 환경에서의 diffusion 모델 커스터마이징 접근성을 넓히는 실질적 효율화다(리뷰어 판단).'
    limitations: '보고된 헤드라인 수치가 FLUX.1-dev(12B) 단일 모델 기준이며(리뷰어 판단), 다른 규모나 아키텍처의 diffusion 모델에 대한 일반화 여부는 abstract 수준에서 확인되지 않는다(논문 내 명시 없음).'
    future_work: '논문 내 명시 없음 — abstract에는 향후 연구 방향이 제시되지 않는다.'
    resources: 'ICML 2026 포스터 페이지가 확인되었으나(발표 확정), 공개 코드 저장소나 체크포인트 링크는 확인되지 않았다 (공개 링크 확인 안 됨).'
  en:
    background: 'Diffusion models have become a dominant paradigm for high-quality generative modeling, and post-training (fine-tuning) is essential for adapting them to diverse downstream applications.'
    problem: 'Post-training of large diffusion models remains challenging due to prohibitive memory footprints and slow training speed, which existing parameter-efficient fine-tuning methods only partially address.'
    prior_limits: 'Not stated in the paper — the abstract does not specify exactly where existing parameter-efficient methods (e.g. standard LoRA) fall short on memory or speed.'
    goal: 'Match standard LoRA fine-tuning quality while dramatically improving the memory and speed of the post-training process itself via native 4-bit computation.'
    method: '**FourTune** is an end-to-end **W4A4G4** (4-bit weights, activations, and gradients) post-training framework. It augments the standard LoRA architecture with a **frozen numerical stabilizer** in a triple-branch hybrid pipeline that isolates quantization-sensitive outliers, enabling stable training under native 4-bit computation. It further uses hardware-efficient block-wise quantization and custom fused kernels to support quantized backpropagation and reduce memory-bandwidth overhead.'
    key_idea: 'Isolating quantization-sensitive outliers into a separate frozen branch lets the trainable LoRA path stay in 4-bit while that branch absorbs the factors that would otherwise make native 4-bit training numerically unstable.'
    validation: 'Evaluated across customization, reinforcement learning, and distillation downstream tasks, applied to the 12B-parameter FLUX.1-dev model and compared against BF16 LoRA.'
    results: 'Matches full-precision fine-tuning quality across all three task types while reducing memory overhead by **2.25×** and increasing end-to-end training throughput by **2.27×** versus BF16 LoRA, on FLUX.1-dev (12B).'
    comparison: 'The primary comparison is against BF16 (full-precision-family) LoRA fine-tuning; no direct numeric comparison against other 4-bit diffusion-model quantization methods (e.g. inference-only PTQ approaches) is given at the abstract level.'
    significance: 'By making the post-training (learning) process itself 4-bit, rather than only inference, this lowers the memory/speed barrier for fine-tuning large diffusion models — a concrete efficiency gain for on-device or resource-constrained diffusion-model customization (reviewer judgment).'
    limitations: 'The headline numbers are reported for a single model, FLUX.1-dev (12B) (reviewer judgment); generalization to other diffusion model scales or architectures is not confirmed at the abstract level (not stated in the paper).'
    future_work: 'Not stated in the paper — the abstract does not describe future work directions.'
    resources: 'An ICML 2026 poster page was verified (confirming acceptance), but no public code repository or checkpoint link could be verified (no public release verified).'
source: "autosweep"
---

## Notes

The framing is worth noting precisely: this is 4-bit *training* (post-training/fine-tuning, W4A4G4 including gradients), not 4-bit inference-only PTQ — a harder numerical-stability problem than the more common inference-quantization line (e.g. SVDQuant/nunchaku for diffusion inference), since gradients must also survive 4-bit precision without diverging.
