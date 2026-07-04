---
title: 'Gated Delta Networks: Improving Mamba2 with Delta Rule'
arxivId: '2412.06464'
authors: 'Songlin Yang, Jan Kautz, Ali Hatamizadeh'
lab: 'MIT / NVIDIA'
venue: 'ICLR 2025'
date: 2026-07-02
tags: [linear-attention, hybrid-architecture, kv-cache]
topic: hybrid-architecture
summary: 'Unifies Mamba2-style gated decay with the delta-rule targeted memory update into a single hardware-efficient linear-attention layer that beats both parents and anchors strong hybrid stacks with sliding-window attention.'
summary_ko: 'Mamba2식 gated decay와 delta rule의 targeted memory update를 하나의 hardware-efficient linear-attention 레이어로 통합해 두 부모 모델을 모두 능가하고, sliding-window attention과의 hybrid 스택의 핵심 레이어가 된 논문.'
links: [minimax-01, duo-attention]
resources:
  - label: arXiv
    url: 'https://arxiv.org/abs/2412.06464'
  - label: PDF
    url: 'https://arxiv.org/pdf/2412.06464'
  - label: GitHub
    url: 'https://github.com/NVlabs/GatedDeltaNet'
analysis:
  ko:
    background: 'Linear attention / state-space 레이어는 길이에 비례해 커지는 KV cache를 고정 크기 matrix state로 대체해 O(1) 메모리 추론을 제공하지만, retrieval과 long-context 성능에서 Transformer에 밀려 왔다.'
    problem: '고정 state의 메모리 관리는 gating(전역적 decay)과 delta rule(특정 key-value association의 정밀 교체)이라는 별개 메커니즘으로 나뉘어 발전해, 오래된 컨텍스트를 빠르게 지우면서 동시에 특정 연관만 정확히 덮어쓰는 능력을 한 레이어가 갖지 못했다.'
    prior_limits: 'Mamba2의 uniform decay는 state 전체를 무차별하게 바래게 해 정밀 retrieval이 약하고(S-NIAH-3 4K에서 **4.6%**), DeltaNet은 rank-1 교체만 가능해 전역 forgetting 수단이 없어 노이즈 있는 retrieval에서 무너진다(S-NIAH-2 8K에서 **14.4%**).'
    goal: 'Gating과 delta rule을 하나의 recurrent update로 통합하되, tensor core에서 학습 가능한 chunkwise-parallel 형태를 유지해 두 부모 모델을 모두 능가하는 것이 목표다.'
    method: 'Gated delta rule S_t = S_{t-1} α_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ — data-dependent 스칼라 게이트 α_t로 state를 decay시킨 뒤 β_t 비율의 rank-1 delta 보정을 적용하며, WY representation + UT transform 기반 chunkwise-parallel 알고리즘(chunk 크기 64)으로 recurrence를 matmul 형태로 학습한다.'
    key_idea: 'Gating과 delta rule은 경쟁이 아니라 상보 관계라는 관찰 — α_t→0이면 급속 memory 소거, α_t→1이면 순수 delta rule로 환원되므로, 스칼라 게이트 하나로 전역 forgetting과 국소 정밀 편집을 같은 레이어에서 얻는다.'
    validation: 'FineWeb-Edu **100B** tokens로 400M/1.3B 모델을 학습(LLaMA-2 tokenizer, 4K seq)해 Wikitext/LAMBADA perplexity, zero-shot commonsense, S-NIAH, recall-intensive 태스크(SWDE·SQuAD·FDA), LongBench 14개 태스크, H100 학습 throughput으로 검증했다.'
    results: '1.3B에서 Wikitext ppl **16.42**(Mamba2 16.56, DeltaNet 17.71), zero-shot 평균 **55.32%**, S-NIAH-2 8K **29.6%**, LongBench 평균 **16.6%**(vs 13.5/13.6)를 기록했고, hybrid H1/H2는 zero-shot **56.4%/56.2%**, recall 평균 **40.1%**까지 끌어올렸다.'
    comparison: 'Transformer++(FlashAttention-2), Mamba2, DeltaNet, Samba류 hybrid와 비교해 품질은 전반적으로 우위이며 H100 학습 throughput은 DeltaNet과 사실상 동일(~48-52K tok/s)해 gating의 추가 비용이 미미함을 보였다.'
    significance: '고정 state 레이어의 retrieval 능력은 state 크기가 아니라 state update의 형태가 결정한다는 것을 보인 결과로, gated delta rule은 이후 Qwen3-Next 같은 production hybrid(full+linear) attention 스택의 표준 linear 컴포넌트로 채택됐다.'
    limitations: '논문 스스로 gating이 완벽한 기억 보존을 희생해 순수 synthetic S-NIAH-1에서는 DeltaNet보다 소폭 낮고, real recall 태스크에서 Mamba2 대비 이득이 synthetic보다 줄어든다고 명시하며, 고정 state의 용량 상한 자체는 hybrid로만 우회될 뿐 해소되지 않는다(리뷰어 판단).'
    future_work: '논문은 GLA식 diagonal gating으로의 확장, negative eigenvalue 허용을 통한 state-tracking 표현력 강화, 20K 초과 시퀀스에서의 검증을 향후 과제로 제시한다.'
    resources: '공식 학습 코드가 NVlabs/GatedDeltaNet GitHub에 공개되어 있고, 커널 구현은 flash-linear-attention(fla-org) 라이브러리에 통합되어 있다.'
  en:
    background: 'Linear-attention / state-space layers replace the length-proportional KV cache with a fixed-size matrix state for O(1)-memory inference, but have trailed Transformers on retrieval and long-context tasks.'
    problem: 'Memory control in fixed-state models evolved as two separate mechanisms — gating (global decay) and the delta rule (precise replacement of a specific key-value association) — so no single layer could both rapidly erase stale context and surgically overwrite targeted associations.'
    prior_limits: "Mamba2's uniform decay fades the entire state indiscriminately, hurting precise retrieval (**4.6%** on S-NIAH-3 at 4K), while DeltaNet performs only rank-1 replacement with no global forgetting and collapses under noisy retrieval (**14.4%** on S-NIAH-2 at 8K)."
    goal: 'Unify gating and the delta rule in one recurrent update while preserving a chunkwise-parallel, tensor-core-friendly training form, so the combined layer surpasses both parents.'
    method: 'The gated delta rule S_t = S_{t-1} α_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ applies a data-dependent scalar decay α_t to the state before a rank-1 delta correction scaled by β_t, trained via a chunkwise-parallel algorithm (chunk size 64) using the WY representation and UT transform to keep the recurrence in matmul form.'
    key_idea: 'Gating and the delta rule are complementary rather than competing — α_t→0 gives rapid memory erasure and α_t→1 recovers the pure delta rule — so a single scalar gate grants one layer both global forgetting and local precise editing.'
    validation: '400M/1.3B models trained on **100B** FineWeb-Edu tokens (LLaMA-2 tokenizer, 4K sequences) are evaluated on Wikitext/LAMBADA perplexity, zero-shot commonsense suites, S-NIAH, recall-intensive tasks (SWDE, SQuAD, FDA), 14 LongBench tasks, and H100 training throughput.'
    results: 'At 1.3B it reaches Wikitext ppl **16.42** (vs Mamba2 16.56, DeltaNet 17.71), **55.32%** average zero-shot accuracy, **29.6%** on S-NIAH-2 at 8K, and **16.6%** LongBench average (vs 13.5/13.6), with hybrids H1/H2 pushing zero-shot to **56.4%/56.2%** and recall average to **40.1%**.'
    comparison: 'Against Transformer++ (FlashAttention-2), Mamba2, DeltaNet, and Samba-style hybrids it wins broadly on quality while matching DeltaNet training throughput on H100 (~48-52K tok/s), showing the gating adds only marginal overhead.'
    significance: 'It demonstrates that the form of the state update — not just state size — determines retrieval ability in constant-memory layers, and the gated delta rule has since been adopted as the standard linear component in production hybrid (full+linear) attention stacks such as Qwen3-Next.'
    limitations: 'The paper itself notes that gating sacrifices perfect memory retention (slightly below pure DeltaNet on synthetic S-NIAH-1) and that gains over Mamba2 shrink on real recall tasks, while the fixed-state capacity ceiling is only bypassed by hybrids, not removed (reviewer judgment).'
    future_work: 'The authors name GLA-style diagonal gating, allowing negative eigenvalues for stronger state tracking, and evaluation beyond 20K-token sequences as future directions.'
    resources: 'Official training code is released at the NVlabs/GatedDeltaNet GitHub repository, and kernel implementations are integrated into the flash-linear-attention (fla-org) library.'
source: 'manual'
---

## Notes

Gated DeltaNet is the cleanest demonstration to date that decay and targeted replacement are complementary axes of memory control in constant-state sequence layers, and its hybrid variants (interleaving sliding-window attention or Mamba2 layers) preview the full+linear attention recipe now common in frontier models. For efficiency work, the interesting boundary is exactly where the fixed-size state fails and the hybrid must fall back to full attention — that seam is where KV-cache cost actually lives in these stacks. The chunkwise WY-representation algorithm is also a good reference point for what "hardware-efficient" must mean for any new recurrence: the update has to reduce to tensor-core matmuls, or the throughput story dies regardless of quality.

Hybrid(full+linear) attention 아키텍처에서 linear 쪽 표준 부품이 된 레이어라, KV cache 관점에서는 "남아 있는 full attention 레이어의 캐시를 어떻게 다루느냐"가 다음 병목이 된다.
