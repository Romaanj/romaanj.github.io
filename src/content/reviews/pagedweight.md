---
title: "PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization"
arxivId: "2607.16184"
authors: "Yuchen Yang, Yifan Zhao, Anisha Dasgupta, Sasa Misailovic"
date: 2026-07-20
tags: ["moe", "quantization", "serving"]
topic: 'serving'
summary: "PagedWeight treats MoE expert weights as elastic, pageable memory — like vLLM's PagedAttention does for the KV cache — dynamically re-quantizing routed experts at runtime to free GPU memory for a growing KV cache, reaching FP16-equivalent accuracy with up to 72.0% memory savings and 1.94x throughput."
summary_ko: "PagedWeight는 vLLM의 PagedAttention이 KV 캐시에 하던 일을 MoE 전문가 가중치에 적용해, 라우팅된 전문가를 런타임에 동적으로 재양자화함으로써 성장하는 KV 캐시를 위한 GPU 메모리를 확보한다 — FP16 동급 정확도를 유지하면서 최대 72.0% 메모리 절감과 1.94배 처리량 향상을 달성한다."
links: ["kronq", "kivi"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.16184' }
figures:
  - src: /figures/pagedweight/fig1.png
    caption: "Static quantization fixes MoE weights at one precision; PagedWeight offloads/restores weight pages at runtime to free headroom for the KV cache."
    caption_ko: "정적 양자화는 MoE 가중치를 하나의 정밀도로 고정하지만, PagedWeight는 런타임에 가중치 페이지를 오프로드/복원해 KV 캐시를 위한 여유 공간을 확보한다."
    credit: "Figure 1 from arXiv:2607.16184 — authors' figure"
analysis:
  ko:
    background: 'Mixture-of-Experts(MoE)는 일부 전문가만 활성화해 연산량을 크게 줄이면서도 정확도를 유지해, 저장소 규모 코딩이나 장문서 분석 같은 장문맥 작업에 특히 매력적이다. 하지만 MoE는 총 파라미터 수를 키워 GPU 메모리를 크게 잠식하며, 논문에 따르면 MoE 모델 로딩만으로 GPU 메모리의 **60% 이상**을 차지할 수 있다 — 이는 문맥 길이와 함께 자라나는 KV 캐시와 같은 메모리를 두고 직접 경쟁한다. vLLM의 PagedAttention은 KV 캐시를 페이지 단위로 관리해 단편화를 줄이고, KV 캐시 압축은 그 자체의 메모리 사용량을 줄이지만, 둘 다 또 다른 메모리 병목인 MoE 가중치에는 손을 대지 못한다.'
    problem: '기존 양자화 기법은 대부분 정적이다 — 모델이 실행되기 전에 가중치를 한 번 수정할 뿐이다. 런타임 정밀도-적응 기법(DP-LLM)은 정밀도가 요청 전체에 고정될 필요가 없음을 보였지만, 그 가변성을 실제 GPU 메모리 재배치에는 쓰지 않는다. 즉 KV 캐시가 커질 때 품질을 해치거나 추론을 중단시키지 않으면서 덜 중요한 전문가 가중치의 메모리를 풀어줄 메커니즘이 존재하지 않는다.'
    prior_limits: 'Any-Precision LLM(APL)은 여러 비트폭을 겹친 bit-plane 포맷으로 표현할 수 있지만 하나의 균일한 정밀도만 적용한다. DP-LLM은 요청별로 정밀도를 적응시키지만 동적 정밀도 프레임워크 특성상 고비트 텐서를 계속 메모리에 유지해야 하고 추론 중 고비트·저비트 경로를 모두 역양자화해야 해, 메모리를 실제로 풀어주지는 못한다. MxMoE는 정적 혼합정밀도를 쓰지만 "가짜 양자화(fake quantization)" 구현이라 실제로는 고정밀도로 저장·실행되며 여러 벌의 가중치 세트를 함께 로드해, 실제 메모리 사용량이 이론적 사용량보다 훨씬 크다.'
    goal: '세 가지 명시적 설계 목표를 만족시키는 것이 목표다 — (1) 동적 KV/가중치 트레이드오프: KV 캐시가 커질 때 가중치가 점유한 GPU 메모리를 풀어줄 것, (2) 품질-인지 전문가 선택: 비트폭 변경이 출력 품질에 가장 적은 영향을 줄 것으로 예측되는 MoE linear-block을 고를 것, (3) 비동기 실행: 페이지 이동이 서빙과 겹쳐 실행되어 추가 비용을 더하지 않을 것.'
    method: '**PagedWeight**는 라우팅된 전문가의 Any-Precision bit-plane과 lookup-table(LUT) 버퍼를 — PagedAttention이 KV 블록을 페이징하듯 — 런타임에 비트폭을 낮추거나 복원할 수 있는 GPU 상주 "페이지"로 취급한다. 구조화된 플래너가 세 신호를 결합해 어떤 페이지를 낮출지 결정한다: 오프라인에서 계산한 Hessian-가중 민감도(전문가 linear-block × 지원 비트폭 전환별), 라우팅-인지 그룹화(자주 라우팅되는 전문가 보호), 프롬프트별 residual 보정. 페이지 오프로드/재로드는 비동기로 실행되어 안전한 실행 경계에서만 커밋되며, Any-Precision bit-plane과 LUT를 직접 읽는 융합 mixed-precision MoE CUDA 커널도 함께 구현했다.'
    key_idea: '핵심은 MoE 전문가 가중치 자체를 — vLLM이 이미 KV 캐시에 적용한 것과 같은 — 페이징 가능하고 신축적인 자원으로 재정의한다는 것이다. 가중치를 고정된 자산으로 보고 KV 캐시만 유연하게 관리하던 기존 구도를 뒤집는다 — 새로 들어온 인기 도서를 위한 서가 공간이 필요할 때, 잘 찾지 않는 책들을 일시적으로 저해상도 마이크로필름으로 보내는 도서관처럼, 하나의 고정된 인쇄본에 묶여 있지 않는 것과 같다.'
    validation: '14.3B~46.7B 규모의 오픈 MoE 모델 3종(Qwen1.5-MoE-A2.7B, Mixtral-8x7B, Gemma-4-26B-A4B)에서 vLLM v0.20.1 위에 구현해 평가한다. 비교 대상은 APL(균일 양자화), MxMoE(정적 혼합정밀도), DP-LLM(동적 혼합정밀도) 세 baseline이다. WikiText2/C4 perplexity, GSM8K/MATH-500 추론 정확도, LongBench 장문맥 태스크(Passage Retrieval, NarrativeQA, QMSum), 생성 처리량, 피크 GPU 메모리를 함께 보고한다.'
    results: 'FP16 동급 정확도를 유지하면서 최대 **72.0%** GPU 메모리 절감과 **1.94배** 처리량 향상을 달성하고, 비슷한 메모리 예산에서 양자화 baseline 대비 최대 **39.3%** 품질 개선을 최대 4.1% 처리량 손실로 얻는다. Qwen1.5-MoE-A2.7B 장문맥 평가(Table 2)에서는 FP16이 35.25GB로 평균 17.0%를 내는 것과 동일한 17.0%를 단 **9.86GB**로 달성하는 반면, APL은 비슷한 메모리에서 12.2%에 그친다.'
    comparison: 'Perplexity, 추론 정확도, 장문맥 태스크 전반에서 APL(균일 양자화)·DP-LLM(정밀도는 적응하지만 메모리를 풀어주지 못함)·MxMoE(이론상 메모리 대비 실제 메모리가 훨씬 큰 "가짜 양자화" 구현) 모두를 일관되게 상회하는 품질-메모리 트레이드오프를 보인다.'
    significance: 'KV 캐시 메모리 관리(PagedAttention, KV 압축)는 이미 성숙한 반면 GPU 메모리의 또 다른 주된 소비자인 MoE 전문가 가중치는 여전히 정적이었다는, MoE 서빙 스택의 실질적 사각지대를 정확히 짚어낸다. 가중치와 KV 캐시를 하나의 "페이징" 추상화 아래 통합하면, 두 자원 중 그 순간 더 필요한 쪽에 메모리를 배분하는 통합 스케줄러로 나아갈 여지가 열린다(리뷰어 판단이 일부 포함되나 저자들의 기여 목록이 이 방향을 뒷받침한다).'
    limitations: '저자들이 Conclusion에서 직접 밝히는 한계는, 특정 양자화 포맷(Any-Precision bit-plane)과 특정 민감도 지표(Hessian-가중)에 한해서만 성능을 시연했다는 것이다 — 다른 양자화 포맷 전반에 대한 일반화는 검증되지 않았다.'
    future_work: '저자들은 두 방향을 명시한다 — (1) 호환되는 레이아웃과 커널을 갖는 다른 양자화 가중치 포맷으로 PagedWeight를 확장하는 것, (2) KV 캐시 압박 하에서 품질-메모리 트레이드오프를 더 개선하기 위해 프롬프트별 전문가 민감도를 추정하는 상보적 방법을 탐구하는 것.'
    resources: '논문 본문·HTML 어디에도 공식 GitHub나 코드 저장소 링크가 없다 — 공개 링크 확인 안 됨. arXiv 초록 페이지만 curl로 200 응답을 확인했다.'
  en:
    background: 'Mixture-of-Experts (MoE) activates only a subset of experts, cutting computation sharply while preserving accuracy, which makes it especially attractive for long-context tasks like repository-scale coding and long-document analysis. But MoE inflates total parameter count and consumes GPU memory heavily — per the paper, loaded MoE model weights alone can occupy over **60%** of GPU memory, competing directly with the KV cache, which grows with context length. vLLM''s PagedAttention pages the KV cache to reduce fragmentation, and KV-cache compression shrinks its own footprint, but neither touches the other major memory bottleneck: MoE weights.'
    problem: 'Most existing quantization is static — it modifies weights once, before the model runs. Runtime precision-adaptation methods (DP-LLM) show precision need not be fixed across a whole request, but they don''t use that variability to actually reallocate GPU memory. There is no mechanism to free memory from less-critical expert weights as the KV cache grows, without hurting quality or interrupting inference.'
    prior_limits: 'Any-Precision LLM (APL) can represent multiple bitwidths in an overlaid bit-plane format but applies only one uniform precision. DP-LLM adapts precision per request, but its dynamic-precision framework must keep higher-bit tensors resident and dequantizes both high- and low-bit paths during inference, so it never actually frees memory. MxMoE performs static mixed-precision quantization, but its "fake quantization" implementation stores and executes weights at higher precision while loading multiple weight sets, so its real memory footprint is much larger than its theoretical one.'
    goal: 'Satisfy three explicit design objectives: (1) dynamic KV/weight tradeoff — release GPU memory held by weights when the KV cache needs to grow; (2) quality-aware expert selection — choose MoE linear-blocks predicted to have the smallest output-quality impact from a bitwidth change; (3) asynchronous execution — page movement must overlap with serving and add no extra cost.'
    method: '**PagedWeight** treats each routed expert''s Any-Precision bit-plane and lookup-table (LUT) buffers as GPU-resident "pages" whose bitwidth can be lowered or restored at runtime — the same way PagedAttention pages KV blocks, but for weight state instead. A structured planner combines three signals to decide which pages to downgrade: offline Hessian-weighted sensitivity scores per expert linear-block and supported bitwidth transition, routing-aware grouping that protects frequently-routed experts, and prompt-specific residual corrections. Page offload/reload runs asynchronously and commits only at safe execution boundaries, alongside a fused mixed-precision MoE CUDA kernel that reads Any-Precision bit-planes and LUTs directly.'
    key_idea: 'The core move is redefining MoE expert weights themselves as a pageable, elastic resource — the same abstraction vLLM already applies to the KV cache — rather than treating weights as fixed and only the KV cache as flexible. It inverts the usual setup where weights are one fixed asset and only the cache is managed dynamically — like a library that temporarily sends rarely-requested books to lower-resolution microfiche when shelf space is needed for popular new arrivals, rather than being locked into one fixed printing per book.'
    validation: 'Evaluated on three open MoE models spanning 14.3B-46.7B (Qwen1.5-MoE-A2.7B, Mixtral-8x7B, Gemma-4-26B-A4B) built on vLLM v0.20.1. Compared against three baselines: APL (uniform quantization), MxMoE (static mixed-precision), and DP-LLM (dynamic mixed-precision). Reports WikiText2/C4 perplexity, GSM8K/MATH-500 reasoning accuracy, LongBench long-context tasks (Passage Retrieval, NarrativeQA, QMSum), generation throughput, and peak GPU memory.'
    results: 'Achieves FP16-equivalent accuracy with up to **72.0%** GPU memory savings and **1.94x** throughput improvement, and improves quality over quantization baselines by up to **39.3%** at a similar memory budget with at most 4.1% throughput loss. On Qwen1.5-MoE-A2.7B long-context evaluation (Table 2), it matches FP16''s 17.0% average LongBench score using only **9.86 GB** versus FP16''s 35.25 GB, while APL reaches only 12.2% at a comparable memory footprint.'
    comparison: 'Consistently better quality-memory tradeoff than APL (uniform quantization), DP-LLM (adapts precision but never frees memory), and MxMoE (whose "fake quantization" implementation has real memory far exceeding its theoretical memory) — across perplexity, reasoning accuracy, and long-context tasks alike.'
    significance: 'Pinpoints a real blind spot in the MoE serving stack: KV-cache memory management (PagedAttention, KV compression) had matured while the other major consumer of GPU memory — MoE expert weights — stayed static. Unifying both under one elastic "paging" abstraction opens the door to a combined memory scheduler that allocates GPU memory to whichever resource, weights or KV cache, needs it most at a given moment (partly reviewer judgment, though supported by the authors'' own contribution framing).'
    limitations: 'The authors state in the Conclusion that they demonstrate performance only on a specific quantization format (Any-Precision bit-planes) and sensitivity metric (Hessian-weighted) — generalization across other quantized weight formats is untested.'
    future_work: 'The authors name two explicit directions: (1) extending PagedWeight to other quantized weight formats with compatible layouts and kernels, and (2) exploring complementary methods for estimating prompt-wise expert sensitivity to further improve the quality-memory tradeoff under KV-cache pressure.'
    resources: 'No official GitHub repository or code release is linked anywhere in the paper or its HTML rendering — no public release verified. Only the arXiv abstract page was confirmed (curl, HTTP 200).'
thread:
  ko: |-
    vLLM의 PagedAttention은 KV 캐시를 페이지 단위 블록으로 관리해 단편화를 줄이고 배칭 효율을 높이는 것을 서빙 시스템의 표준으로 만들었다. 이와 나란히, 별개의 계보로 양자화 연구(Any-Precision LLM, DP-LLM, MxMoE)가 가중치 메모리 문제를 다뤄왔다 — 대부분 KV 캐시에서 그 순간 무슨 일이 벌어지는지와 무관하게, 한 번 적용되는 정적인 변환으로서.

    PagedWeight의 전환은 KV 캐시 관리를 혁신했던 바로 그 페이징 추상화를 가중치 메모리 자체에 적용한다는 데 있다 — Any-Precision bit-plane을 고정된 자산이 아니라 신축적이고 교체 가능한 페이지로 취급하고, 그 페이징 결정을 실시간 KV 캐시 압박과 연결한다(Hessian-가중 + 라우팅-인지 + 프롬프트 residual로 구성된 품질-인지 플래너를 통해). "가중치를 한 번 양자화하고 KV는 따로 관리한다"는 구도에서, 하나의 정책이 두 자원을 실시간으로 함께 조정하는 구도로의 이동이다.

    가중치 메모리가 KV 메모리처럼 신축적이 되고 나면, 자연스러운 다음 질문은 진짜 통합 메모리 스케줄러다 — "가중치는 한 번, KV는 별도로"가 아니라, 요청마다 가중치와 KV 캐시 사이에 하나의 GPU 메모리 예산을 동적으로 배분하는 단일 정책, 그리고 오늘의 특정 Any-Precision 포맷을 넘어 다른 양자화 레이아웃 전반으로의 일반화다. 저자들 스스로도 future work에서 바로 이 두 방향 — 다른 양자화 포맷으로의 확장과 더 나은 프롬프트별 민감도 추정 — 을 명시한다.
  en: |-
    vLLM's PagedAttention made page-block management of the KV cache the standard way to fight fragmentation and boost batching efficiency in serving systems. Running alongside it, a separate lineage of quantization work (Any-Precision LLM, DP-LLM, MxMoE) tackled weight memory — mostly as a static transformation applied once, independent of whatever was happening in the KV cache at the time.

    PagedWeight's shift is to apply the very paging abstraction that revolutionized KV-cache management to weight memory itself — treating Any-Precision bit-planes as elastic, swappable pages rather than a fixed asset, and coupling that paging decision to real-time KV-cache pressure through a quality-aware planner (Hessian-weighted sensitivity + routing-aware grouping + prompt-specific residuals). It's a move from "quantize the weights once, manage the KV cache separately" to one policy that coordinates both resources together, in real time.

    Once weight memory is elastic like KV memory, the natural next question is a genuinely unified memory scheduler — not "weights once, KV separately," but a single policy that dynamically splits one GPU memory budget between weights and KV cache per request, generalized beyond today's specific Any-Precision format to quantization layouts more broadly. The authors themselves name exactly these two directions as future work: extending to other quantized formats, and better prompt-wise sensitivity estimation.
sparks:
  - ko: '저자들이 명시한 future work — 호환 가능한 레이아웃과 커널을 갖는 다른 양자화 포맷(예: GPTQ류 정수 양자화)으로 PagedWeight의 페이징 추상화를 확장하면 Any-Precision 특유의 이득이 얼마나 유지될까?'
    en: "The authors' own stated future work — if PagedWeight's paging abstraction were extended to other quantized formats with compatible layouts and kernels (e.g. GPTQ-style integer quantization), how much of the Any-Precision-specific gain would carry over?"
  - ko: '저자들이 명시한 두 번째 future work — 프롬프트별 전문가 민감도를 추정하는 상보적 방법을 찾는 것 — 은 지금의 Hessian-가중 오프라인 지표가 프롬프트마다 얼마나 다른 전문가를 "중요"하다고 판단하는지에 대한 열린 질문을 남긴다.'
    en: "The authors' second stated future-work direction — finding complementary methods for prompt-wise expert-sensitivity estimation — leaves open just how differently the current Hessian-weighted offline metric would judge 'importance' from one prompt to the next."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
