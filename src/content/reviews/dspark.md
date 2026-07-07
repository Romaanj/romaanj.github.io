---
title: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
arxivId: "2607.05147"
authors: "Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, et al."
lab: "DeepSeek"
date: 2026-07-07
tags: ["speculative-decoding", "serving"]
topic: 'serving'
summary: "A semi-autoregressive drafter combined with confidence-scheduled verification fixes the intra-block acceptance decay of parallel speculative-decoding drafters, delivering 51-85% throughput and latency gains in DeepSeek-V4 production serving."
summary_ko: "반-자기회귀적 drafter와 confidence 기반 검증 스케줄링을 결합해 parallel 추론 drafter의 block 내 수락률 감소 문제를 해결하고, DeepSeek-V4 실제 서빙에서 51-85%의 처리량·지연 개선을 달성한다."
links: ["deepseek-r1"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.05147' }
  - { label: 'GitHub', url: 'https://github.com/deepseek-ai/DeepSpec' }
  - { label: 'Hugging Face checkpoints', url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark' }
analysis:
  ko:
    background: 'Speculative decoding은 경량 drafter가 여러 token을 제안하고 target 모델이 한 번에 검증하는 방식으로 LLM 추론을 가속하며, 최근의 parallel(non-autoregressive) drafter는 한 번의 forward pass로 긴 시퀀스를 제안할 수 있다.'
    problem: 'Parallel drafter는 token 간 의존성을 모델링하지 않기 때문에 drafted block 뒷부분으로 갈수록 수락률이 급격히 감소하고, 이 감소를 무시한 채 긴 block 전체를 검증하면 거부될 가능성이 높은 token에 검증·배치 용량을 낭비하게 된다.'
    prior_limits: '순수 autoregressive drafter(Eagle류)는 token 간 의존성을 모델링해 block 전반의 수락률을 안정적으로 유지하지만 초안 생성이 느리고, 순수 parallel drafter는 빠르지만 block 내에서 빠르게 감소하며, 둘 다 요청별 실제 수락 가능성에 따라 검증 길이를 조정하지 않는다.'
    goal: '병렬 생성의 초안 생성 속도를 대부분 유지하면서 의존성 인식 drafting의 수락 길이 이점을 회복하고, 실제 서빙 부하 하에서 요청별로 검증 비용을 적응시키는 것.'
    method: 'DSpark는 block 내 의존성을 위한 경량 순차 모듈("Markov head")을 결합한 parallel backbone인 반-자기회귀적 drafter를 사용하며, token별 confidence head가 접두사 **생존 확률**을 추정하고 시스템은 측정된 처리량을 최대화하는 요청별 검증 길이를 선택하는 confidence 기반 검증 스케줄링과 결합한다.'
    key_idea: '학습된 confidence head가 각 draft token이 target 검증에서 살아남을 확률을 예측하고(분석적인 total-variation-distance 수락률로 지도학습), 이 확률들을 drafted 시퀀스를 따라 곱하면 접두사 생존 곡선이 나오는데, 스케줄러는 이를 이용해 생존율이 낮은 요청은 검증을 짧게, 높은 요청은 길게 조정한다.'
    validation: 'Qwen3-4B/8B/14B와 Gemma4-12B에서 autoregressive(Eagle3) 및 parallel(DFlash) drafter와 여러 벤치마크 도메인에 걸쳐 오프라인으로 비교했고, DeepSeek-V4 실서비스(V4-Flash, V4-Pro)에 배포해 실사용자 트래픽 하에서 production MTP-1 베이스라인과 비교했다.'
    results: '오프라인 수락 길이는 테스트한 Qwen3 크기들에서 Eagle3 대비 **26.7-30.9%**, DFlash 대비 **16.3-18.4%** 개선되며, DeepSeek-V4 실서비스에서는 종합 처리량이 V4-Flash(80 tok/s/user SLA)에서 **51%**, V4-Pro(35 tok/s/user SLA)에서 **52%** 개선되고, 동일 처리량 기준 사용자별 생성 속도는 **60-85%** 빨라진다.'
    comparison: '오프라인에서는 Eagle3(autoregressive)와 DFlash(parallel) drafter와, 실서비스에서는 production MTP-1 베이스라인과 비교했다. 논문이 보고하는 수락률 곡선은 전체 문맥 길이가 아니라 block 내 token 위치에 대한 함수다.'
    significance: '오프라인 벤치마크뿐 아니라 실제 프로덕션 규모(대형 랩의 실서비스 시스템)에서 검증된 드문 speculative decoding 논문이며, 엄격한 사용자별 지연 SLA 하에서 적응적·요청별 검증 길이가 처리량에 실용적인 지렛대임을 보여준다.'
    limitations: '저자들은 원래 수락률이 낮은 복잡한 질의의 경우 초기 drafting 연산 비용을 회수할 수 없다고 명시하며, 처리량 모델링은 문맥 길이가 100만 token 같은 극단적인 값보다 훨씬 낮게 유지된다고 가정한다.'
    future_work: '논문은 수락률이 낮은 질의에 대한 난이도 인식 조기 종료(early exiting)를 향후 방향으로 제시한다.'
    resources: 'Eagle3·DFlash·DSpark 구현을 포함한 코드가 GitHub의 DeepSpec 저장소로, DeepSeek-V4-Pro용 DSpark 체크포인트가 Hugging Face에 공개되어 있다.'
  en:
    background: 'Speculative decoding accelerates LLM inference by having a lightweight drafter propose multiple tokens that the target model verifies in one pass, and recent parallel (non-autoregressive) drafters can propose long sequences in a single forward call.'
    problem: 'Parallel drafters suffer rapid acceptance decay toward the end of a drafted block because they lack inter-token dependency modeling, and verifying an entire long block regardless of this decay wastes verification and batch capacity on tokens likely to be rejected.'
    prior_limits: 'Purely autoregressive drafters (Eagle-style) model inter-token dependency and keep acceptance stable across a block but are slower to draft; purely parallel drafters draft fast but decay quickly within a block, and neither adapts verification length to the actual accept-likelihood of a request.'
    goal: 'Recover the accepted-length benefits of dependency-aware drafting while retaining most of the drafting speed of parallel generation, and adapt verification cost per request under production serving load.'
    method: 'DSpark uses a semi-autoregressive drafter — a parallel backbone coupled with a lightweight sequential module (a low-rank "Markov head") for intra-block dependency — together with confidence-scheduled verification, where a per-token confidence head estimates prefix **survival probability** and the system picks a per-request verification length that maximizes measured system throughput.'
    key_idea: 'A trained confidence head predicts the probability that each draft token will survive target verification (supervised by the analytic total-variation-distance acceptance rate), and multiplying these probabilities along the drafted sequence gives a prefix survival curve that the scheduler uses to cut verification short for low-survival requests and extend it for high-survival ones.'
    validation: 'Evaluated offline on Qwen3-4B/8B/14B and Gemma4-12B against autoregressive (Eagle3) and parallel (DFlash) drafters across multiple benchmark domains, and deployed in DeepSeek-V4 production serving (V4-Flash and V4-Pro) under live user traffic against a production MTP-1 baseline.'
    results: 'Offline accepted length improves **26.7-30.9%** over Eagle3 and **16.3-18.4%** over DFlash across the tested Qwen3 sizes; in DeepSeek-V4 production, DSpark improves aggregate throughput **51%** (V4-Flash, 80 tok/s/user SLA) and **52%** (V4-Pro, 35 tok/s/user SLA), and accelerates per-user generation **60-85%** at matched throughput levels.'
    comparison: 'Compared against Eagle3 (autoregressive) and DFlash (parallel) drafters offline, and against a production MTP-1 baseline in the live DeepSeek-V4 deployment; the paper reports acceptance-rate curves as a function of intra-block token position, not as a function of total context length.'
    significance: 'It is a rare speculative-decoding paper validated at real production scale (a major lab live serving system) rather than only offline benchmarks, and demonstrates that adaptive, per-request verification length is a practical lever for throughput under strict per-user latency SLAs.'
    limitations: 'The authors state that for complex queries with inherently low acceptance rates, the upfront drafting compute is unrecoverable, and their throughput modeling assumes context lengths remain well below extremes such as 1M tokens.'
    future_work: 'The paper suggests difficulty-aware early exiting for low-acceptance queries as a direction for future work.'
    resources: 'Code (including Eagle3, DFlash, and DSpark implementations) is released as the DeepSpec repository on GitHub, and DSpark checkpoints for DeepSeek-V4-Pro are released on Hugging Face.'
source: "autosweep"
---

## Notes

The most quotable distinction here is that DSpark's reported acceptance curve is a function of position-within-a-drafted-block, not a function of total conversation/context length — the two are easy to conflate but describe different phenomena (missing intra-block token dependency vs. long-context degradation). Read the 60-85% production numbers as evidence that adaptive verification length is a real, deployable lever, not evidence about long-context acceptance behavior.
