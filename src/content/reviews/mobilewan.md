---
title: "MobileWan: Closing the Quality Gap for Mobile Video Diffusion"
arxivId: "2607.06173"
authors: "Mohsen Ghafoorian, Denis Korzhenkov, Adil Karjauv, Ioannis Lelekas, Noor Fathima, Spyridon Stasis, Hanno Ackermann, Boris van Breugel, Markus Nagel, Fatih Porikli, Animesh Karnewar, Amirhossein Habibian"
lab: "Qualcomm AI Research"
date: 2026-07-09
tags: ["video-diffusion", "on-device", "mobile", "distillation"]
topic: 'on-device'
summary: "A 5B-parameter video diffusion transformer is converted into a constant-memory, chunk-wise autoregressive RNN-like process via recurrence distillation and causal linear attention, becoming the first 5B-scale video diffusion model deployable on a commercial mobile device."
summary_ko: "5B 파라미터 비디오 diffusion transformer를 recurrence distillation과 causal linear attention으로 constant-memory의 chunk-wise autoregressive RNN 형태로 변환해, 상용 모바일 기기에 배포 가능한 최초의 5B 규모 비디오 diffusion 모델을 만들었다."
links: ["mobilellm"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.06173' }
analysis:
  ko:
    background: '비디오 diffusion 모델은 트랜스포머 아키텍처를 수십억 파라미터로 스케일링하며 화질과 모션 일관성을 크게 향상시켜 왔지만, 기존 모바일용 비디오 diffusion 모델은 **0.4-1.8B** 규모의 상대적으로 작은 파라미터 예산에 머물러 있다.'
    problem: '작은 모델 예산이 생성 품질을 제한하는데, 메모리가 제한된 모바일 하드웨어에서 서버급 대형 비디오 diffusion transformer를 효율적으로 배포할 방법이 없다는 것이 문제다.'
    prior_limits: '논문 내 명시 없음 — abstract 수준에서는 기존 0.4-1.8B 모바일 비디오 diffusion 모델들이 왜 그 규모에 머물렀는지에 대한 구체적 기술적 근거(구조/학습 방식)는 제시되지 않는다.'
    goal: '작은 모델을 새로 만드는 대신, 서버급 5B 파라미터 비디오 diffusion transformer(Wan2.2-5B)를 메모리 제약이 있는 모바일 하드웨어에 효율적으로 배포하는 것.'
    method: 'Wan2.2-5B에서 출발해 **recurrence distillation** 프레임워크로 비디오 생성을 constant-memory attention을 갖는 chunk-wise autoregressive 과정으로 변환하고, causal linear attention과 결합해 추론 시 모델이 RNN처럼 동작하면서도 청크 간 시간적 일관성을 유지하게 한다. 여기에 노이즈 편향 희소성 목적함수로 end-to-end 최적화되는 학습형 attention head pruning(head별 binary gate), sampling-step distillation, 메모리 최적화 VAE 디코딩을 결합한다.'
    key_idea: '모델을 축소하는 대신, 크고 bidirectional한 비디오 diffusion transformer를 recurrence distillation과 causal linear attention을 통해 constant-memory RNN류 프로세스로 재정식화하면 동일한 5B 모델을 모바일 메모리 예산 안에서 그대로 구동할 수 있다.'
    validation: 'Wan2.2-5B를 시작점으로 삼아 VBench로 평가했고, 상용 모바일 기기에서 실제 480x832 해상도 5초 분량 비디오 생성의 end-to-end 지연시간을 측정했다.'
    results: '상용 모바일 기기에 배포 가능한 최초의 **5B 규모** 비디오 diffusion 모델이 되었으며, **480x832** 해상도의 **16 FPS**, 5초 분량 비디오를 **20초**의 end-to-end 지연시간으로 생성하고 VBench 점수 **83.79**로 모바일 비디오 생성의 새로운 state-of-the-art를 세웠다.'
    comparison: '기존 0.4-1.8B 규모 모바일 비디오 diffusion 모델들과 비교해 모델을 축소하지 않고도 품질 격차를 좁혔다고 주장하나, 그 모델들과의 구체적 수치 비교표는 abstract 수준에서는 제시되지 않는다(논문 내 명시 없음).'
    significance: 'On-device 생성형 AI 관점에서, recurrence distillation과 구조적 attention-head pruning의 조합이 모델을 축소하지 않고도 서버급 생성 품질을 메모리 제약 하드웨어에서 보존할 수 있음을 보여주며, 이는 비디오를 넘어 온디바이스 생성 모델 압축 전반에 시사점을 준다(리뷰어 판단).'
    limitations: '20초의 end-to-end 지연시간은 아직 실시간 생성이 아니며(리뷰어 판단), 평가가 단일 상용 모바일 기기 기준으로만 보고되어 다양한 기기·배터리/발열 조건에서의 일반화는 abstract 수준에서 확인되지 않는다(리뷰어 판단).'
    future_work: '논문 내 명시 없음 — abstract에는 향후 연구 방향이 제시되지 않는다.'
    resources: '논문에 프로젝트 페이지 URL이 명시되어 있으나 확인 시점에 공개 접근이 되지 않았고, 코드나 체크포인트에 대한 공개 링크는 확인되지 않았다 (공개 링크 확인 안 됨).'
  en:
    background: 'Video diffusion models have improved visual fidelity and motion coherence by scaling transformer architectures to billions of parameters, but existing mobile video diffusion models remain limited to relatively small parameter budgets, typically **0.4-1.8B**.'
    problem: 'Small model budgets restrict generation quality, and there has been no way to efficiently deploy a server-scale, large video diffusion transformer on memory-constrained mobile hardware.'
    prior_limits: 'Not stated in the paper — the abstract does not give specific technical reasons (architecture or training choices) for why prior mobile video diffusion models stayed in the 0.4-1.8B range.'
    goal: 'Instead of training a new small model, efficiently deploy a server-scale **5B**-parameter video diffusion transformer (Wan2.2-5B) on memory-constrained mobile hardware.'
    method: 'Starting from Wan2.2-5B, a **recurrence distillation** framework converts video generation into a chunk-wise autoregressive process with constant-memory attention, combined with causal linear attention so the model operates as an RNN at inference while preserving temporal coherence across chunks. This is combined with a learnable attention-head pruning method (binary per-head gates optimized end-to-end with a noise-biased sparsity objective), sampling-step distillation, and memory-optimized VAE decoding.'
    key_idea: 'Rather than shrinking the model, reformulating a large bidirectional video diffusion transformer into a constant-memory, RNN-like process via recurrence distillation and causal linear attention lets the same 5B model run within a mobile memory budget.'
    validation: 'Evaluated with VBench starting from Wan2.2-5B, and measured end-to-end latency for real 480x832-resolution, 5-second video generation on a commercial mobile device.'
    results: 'Becomes the first **5B-scale** video diffusion model deployable on a commercial mobile device, generating **480x832**, **16 FPS**, 5-second videos in **20 seconds** end-to-end latency, with a VBench score of **83.79** — a new state of the art for mobile video generation.'
    comparison: 'Claims to close the quality gap versus existing 0.4-1.8B mobile video diffusion models without shrinking the model, but a specific numeric comparison table against those models is not given at the abstract level (not stated in the paper).'
    significance: 'From an on-device generative-AI angle, it shows that recurrence distillation combined with structured attention-head pruning can preserve server-scale generation quality on memory-constrained hardware without shrinking the model — a finding relevant beyond video to on-device generative-model compression generally (reviewer judgment).'
    limitations: 'The 20-second end-to-end latency is not yet real-time generation (reviewer judgment), and evaluation is reported on a single commercial mobile device, so generalization across devices and thermal/battery conditions is not established at the abstract level (reviewer judgment).'
    future_work: 'Not stated in the paper — the abstract does not describe future work directions.'
    resources: 'The paper states a project page URL, but it did not resolve to a live public page at verification time, and no public code or checkpoint link could be verified (no public release verified).'
source: "autosweep"
---

## Notes

The core move here is refusing the usual mobile-deployment trade (shrink the model) and instead
changing the *computational form* of a large model (bidirectional diffusion transformer →
constant-memory recurrent process via recurrence distillation + causal linear attention). That
reformulation-over-shrinking approach is the more broadly reusable idea, independent of the video
domain specifics.
