---
title: "Sangam: Efficiently Serving Diffusion LLMs with the AR Stack"
arxivId: "2607.04206"
authors: "Nitin Kedia, Saurabh Agarwal, Myungjin Lee, Aditya Akella"
date: 2026-07-07
tags: ["dllm", "serving", "kv-cache"]
topic: 'serving'
summary: "A dedicated serving system for cached diffusion-LLM inference introduces a deficit token-budget scheduler that manages prefill/decode interference without AR-style chunked prefill, sustaining 3x the query rate of an in-system Fast-dLLM baseline on LLaDA-8B and Dream-7B."
summary_ko: "캐시된 diffusion LLM 추론을 위한 전용 서빙 시스템이 AR 방식 chunked prefill 없이 prefill/decode 간섭을 관리하는 deficit token-budget 스케줄러를 도입해, LLaDA-8B와 Dream-7B에서 in-system Fast-dLLM 베이스라인 대비 3배의 QPS를 유지한다."
links: ["fast-dllm"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.04206' }
  - { label: 'GitHub', url: 'https://github.com/UT-InfraAI/sangam' }
analysis:
  ko:
    background: 'LLaDA, Dream 같은 diffusion LLM(dLLM)은 반복적 denoising으로 텍스트를 생성하며 한 번의 모델 호출로 여러 출력 위치를 동시에 확정할 수 있지만, bidirectional attention 때문에 정확한 autoregressive 방식 KV 캐싱을 쓸 수 없다.'
    problem: 'Fast-dLLM, dKV-Cache 같은 근사 캐싱 기법은 KV activation을 주기적으로 갱신하는데, 이는 서빙 시점에 반복되는 prefill/decode 구조를 재도입한다. 그러나 이 간섭에 대한 AR의 표준 해법인 chunked prefill은 bidirectional block의 일부만 확정하면 나머지의 KV가 무효화되므로 적용할 수 없다.'
    prior_limits: 'Fast-dLLM 등 기존 dLLM 가속 연구는 디코딩 알고리즘 자체를 최적화할 뿐, 캐시된 dLLM 추론에서 서버가 동시 prefill/decode 요청을 어떻게 스케줄링해야 하는지는 다루지 않는다.'
    goal: 'AR 방식의 chunked prefill에 의존하지 않고, 실제 멀티-GPU 하드웨어에서 캐시된 dLLM 추론에 대해 stall 없는 스케줄링을 달성하는 서빙 시스템을 만드는 것.'
    method: 'Sangam은 진행 중인 decode를 항상 먼저 수용하고, 대기 중인 prefill은 누적된 반복당 token budget이 허용하는 동안에만 전체 단위로 수용하며, 남은 budget을 다음 반복으로 이월하는 **deficit token-budget 스케줄러**를 도입하고, prefill이 많은 부하에서는 decode worker로 prefill을 오버플로우시키는 hybrid 모드를 추가한다.'
    key_idea: 'dLLM prefill은 스케줄링 반복 사이에서 분할할 수 없으므로, 토큰 단위 chunking 대신 롤링 token budget으로 페이싱되는 전부-아니면-전무 방식의 admission으로 처리한다.'
    validation: 'LLaDA-8B-Instruct(ShareGPT, decode-heavy)와 Dream-7B-Instruct(arXiv 요약, prefill-heavy)를 단일 **8×H100 80GB** 노드에서 평가했고, colocated·disaggregated·hybrid 서빙 방식을 재현한 Fast-dLLM 스타일 베이스라인과 비교했다.'
    results: 'Colocated 서빙은 decode-heavy 워크로드에서 hybrid 대비 평균 end-to-end 지연을 **9-20%** 줄이고, hybrid는 prefill-heavy 워크로드에서 colocated 대비 **8-20%** 줄이며, colocated Sangam은 동일 지연 기준 in-system Fast-dLLM 베이스라인 대비 **3배**의 QPS(1.0 vs 0.3)를 유지한다.'
    comparison: '논문은 colocated·disaggregated·hybrid 서빙 전략들을 서로, 그리고 재현한 Fast-dLLM 베이스라인과 비교하지만, vLLM 같은 표준 AR 서빙 스택 위의 autoregressive 모델과의 동일 품질 처리량 비교는 보고하지 않는다.'
    significance: 'Diffusion LLM 서빙을 1급 스케줄링 문제로 다룬 초기 시스템 논문 중 하나로, 캐시된 dLLM 서빙에서도 AR 방식의 prefill/decode 간섭이 재현되며 이를 위해서는 AR용을 재사용하는 대신 dLLM 전용 스케줄러가 필요함을 보인다.'
    limitations: '평가가 단일 8-GPU 노드로 한정되어 있고 노드 간 대역폭 영향은 실측하지 않았다고 논문 스스로 명시하며, 순수 지연/처리량 시스템 연구이므로 정확도나 생성 품질 수치는 보고하지 않는다(리뷰어 판단: 품질 비교는 이 논문의 범위 밖이다).'
    future_work: '논문은 계층별 KV 스트리밍과 KV 전송을 decode 연산과 오버랩하는 것을 실현 가능하지만 아직 구현하지 않은 최적화로 언급한다(리뷰어 판단: AR 서빙 스택과의 동일 품질 처리량 비교도 자연스러운 다음 단계다).'
    resources: '코드는 GitHub에 "sangam" 저장소로 공개되어 있으며, Sangam은 기존 LLaDA/Dream 모델 위의 서빙 레이어이므로 별도의 모델 체크포인트는 필요하지 않다.'
  en:
    background: 'Diffusion LLMs (dLLMs) such as LLaDA and Dream generate text through iterative denoising and can commit multiple output positions per model call, but their bidirectional attention prevents exact autoregressive-style KV caching.'
    problem: 'Approximate caching schemes like Fast-dLLM and dKV-Cache periodically refresh KV activations, which reintroduces a repeating prefill/decode structure at serving time, yet the standard AR fix for that interference — chunked prefill — cannot be applied because committing part of a bidirectional block invalidates KV for the rest of it.'
    prior_limits: 'Prior dLLM acceleration work such as Fast-dLLM optimizes the decoding algorithm itself but does not address how a server should schedule concurrent prefill and decode requests for cached dLLM inference.'
    goal: 'Build a serving system that achieves stall-free scheduling for cached dLLM inference on real multi-GPU hardware without relying on autoregressive-style chunked prefill.'
    method: 'Sangam introduces a **deficit token-budget scheduler** that always admits in-flight decodes first, admits waiting prefills only whole and only while an accumulated per-iteration token budget allows, carries any unused budget forward, and adds a hybrid mode that overflows prefills onto decode workers under prefill-heavy load.'
    key_idea: 'Because a dLLM prefill cannot be split across scheduling iterations, admission is treated as an all-or-nothing decision paced by a rolling token budget rather than token-level chunking.'
    validation: 'Evaluated on LLaDA-8B-Instruct (ShareGPT, decode-heavy) and Dream-7B-Instruct (arXiv summarization, prefill-heavy) on a single **8×H100 80GB** node, comparing colocated, disaggregated, and hybrid serving against a reproduced Fast-dLLM-style baseline.'
    results: 'Colocated serving cuts mean end-to-end latency **9-20%** versus hybrid on the decode-heavy workload, hybrid cuts it **8-20%** versus colocated on the prefill-heavy workload, and colocated Sangam sustains **3x** the query rate of an in-system Fast-dLLM baseline at matched latency (QPS 1.0 vs. 0.3).'
    comparison: 'The paper compares colocated, disaggregated, and hybrid serving strategies against each other and against a reproduced Fast-dLLM baseline, but does not report a matched-quality throughput comparison against an autoregressive model on a standard AR serving stack such as vLLM.'
    significance: 'It is among the first systems papers to treat diffusion-LLM serving as a first-class scheduling problem, showing that AR-style prefill/decode interference reappears under cached dLLM serving and needs a dLLM-specific scheduler rather than a reused AR one.'
    limitations: 'The evaluation covers only a single 8-GPU node and the paper states inter-node bandwidth effects were not empirically measured; it also reports no accuracy or generation-quality numbers, since it is a pure latency/throughput systems study (reviewer judgment: quality comparison is outside this paper''s scope).'
    future_work: 'The paper names layerwise KV streaming and overlapping KV transfer with decode computation as feasible but unimplemented optimizations (reviewer judgment: a matched-quality throughput comparison against an AR serving stack is a natural next step).'
    resources: 'Code is released on GitHub as the "sangam" repository; no separate model checkpoints are required since Sangam is a serving layer on top of existing LLaDA/Dream models.'
source: "autosweep"
---

## Notes

Sangam is a systems paper, not an algorithmic one — its contribution is a scheduler, not a new caching or quantization method. The interesting design constraint it surfaces is that bidirectional attention makes dLLM prefill fundamentally atomic (unsplittable), which rules out AR's chunked-prefill trick and forces a different scheduling primitive (deficit token-budget admission). Worth watching for a future paper that closes the gap this one leaves open: a matched-quality throughput comparison between cached dLLM serving and a standard AR stack.
