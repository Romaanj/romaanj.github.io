---
title: "VarRate: Training-Free Variable-Rate KV Cache Compression for Long-Context LLMs"
arxivId: "2607.15498"
authors: "Shahrzad Esmat, Dhawal Shah, Ali Jannesari"
date: 2026-07-20
tags: ["kv-cache", "low-rank-compression", "water-filling"]
topic: 'kv-cache'
summary: "VarRate is a training-free KV cache codec that allocates a per-token low-rank budget via water-filling over query salience instead of evicting tokens, staying within 3.5-5.5 points of uncompressed accuracy under query-agnostic cache reuse where query-aware eviction loses 11-15 points."
summary_ko: "VarRate는 토큰을 삭제하는 대신 쿼리 salience에 대한 water-filling으로 토큰별 저랭크 예산을 배분하는 학습-불필요 KV 캐시 코덱으로, query-agnostic 캐시 재사용 상황에서 query-aware eviction이 11-15점을 잃는 반면 3.5-5.5점 손실에 그친다."
links: ["kivi", "kvquant"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.15498' }
figures:
  - src: /figures/varrate/fig1.png
    caption: "Token selection evicts low-scoring tokens (irreversible); uniform low-rank coding spends equal rank on every token; VarRate varies rank with salience while holding every token above a nonzero floor."
    caption_ko: "토큰 선택(eviction)은 낮은 점수 토큰을 삭제한다(비가역적); 균일 저랭크 코딩은 모든 토큰에 동일한 랭크를 쓴다; VarRate는 salience에 따라 랭크를 가변 배분하면서 모든 토큰을 0이 아닌 하한 위에 유지한다."
    credit: "Figure 1 from arXiv:2607.15498 — authors' figure"
analysis:
  ko:
    background: 'KV(key-value) 캐시는 장문맥 LLM 추론의 주된 메모리 병목이다. 학습이 필요 없는 압축 기법은 크게 두 갈래로 갈린다 — 토큰 선택(eviction, SnapKV/Ada-KV)은 현재 쿼리에 대한 attention 관측 윈도우로 각 토큰을 채점하고 점수가 낮은 토큰을 버리며, 균일 저랭크 코딩(Palu 계열)은 모든 토큰을 유지하되 동일한 축소 랭크로 투영한다. 정밀도를 낮추는 양자화(KIVI, KVQuant, QJL)는 이 둘과 독립된 세 번째 축이다.'
    problem: '두 주류 기법 모두 "한 번 압축해 여러 쿼리에 재사용"하는 query-agnostic reuse(문서 캐시 재사용, prefix caching, 멀티턴 대화) 상황에서 무너진다. Eviction의 salience 점수는 본질적으로 "지금 이 쿼리가 필요로 하는 것"이므로, 압축 시점의 쿼리가 이후 쿼리와 다르면 신호가 낡고, 삭제는 비가역적이므로 정확도가 11-15점 폭락한다. 균일 저랭크 코딩은 애초에 쿼리에 의존하지 않아 낡을 일이 없지만, 랭크 예산을 아무 데나 균등하게 써 절대 중요해지지 않을 토큰에도 중요해질 토큰과 같은 랭크를 쓴다.'
    prior_limits: 'KVzip은 쿼리에 무관한 재구성 기반 점수를 써 query-agnostic reuse에 강건하지만, 그 점수를 얻으려면 전체 모델로 컨텍스트 전체를 여러 번 재인코딩해야 해 절약하려던 prefill 비용의 몇 배를 치른다. 토큰별 랭크를 가변화하려는 시도도 있었지만, DynaKV는 게이트를 미세조정으로 학습해야 하고 OjaKV는 학습이 필요 없는 대신 실제 salience가 아니라 고정된 위치(anchor 토큰 vs 나머지)로 랭크를 나눈다.'
    goal: '토큰을 절대 삭제하지 않으면서도, eviction이 이미 계산하고 있는 것과 같은 값싼 쿼리 salience 신호만으로 — 추가 학습 없이 — 랭크 예산을 정말 중요한 곳에 쓰는 것이 목표다.'
    method: '**VarRate**는 water-filling으로 랭크를 배분한다: r_t = clip(r_min + λ·ŝ_t, r_min, R) — 예산 B가 채워질 때까지 "수위" λ를 올리고, 랭크 상한 R에 닿은 토큰은 클리핑해 잉여를 나머지에 재분배한다(Algorithm 1의 표준 water-filling 반복). salience ŝ_t는 SnapKV 스타일로 짧은 관측 윈도우에 대한 쿼리 attention 점수다. 0이 아닌 하한 r_min 덕분에 어떤 토큰도 랭크 0(완전 삭제)으로 떨어지지 않는다 — 잘못 판단된 토큰은 거칠어질 뿐, 사라지지 않는다.'
    key_idea: '두 주류 기법의 실패는 사실 독립적인 두 선택 — "중요도를 어떻게 채점하는가"와 "낮은 점수에 어떻게 반응하는가" — 이 우연히 나쁘게 결합된 결과라는 관찰이 핵심이다. VarRate는 eviction의 값싼 채점 방식은 그대로 두고, 반응만 파괴적(삭제)에서 가역적(거칠게 하기)으로 바꾼다 — 저장 공간이 부족할 때 사진을 지우는 대신 썸네일 해상도만 슬쩍 낮추는 것과 같다. 이 하나의 교체가 낡은 신호를 치명적인 것에서 감당 가능한 것으로 바꾼다.'
    validation: 'Llama-3.1-8B와 Qwen2.5-7B에서 LongBench 16개 태스크로 평가한다. 매칭된 메모리 예산에서 자체 균일-랭크 ablation, 발표된 저랭크 코덱 Palu, 양자화 baseline, 그리고 query-agnostic reuse 전용 방법인 KVzip과 비교하며, 압축을 한 번 하고 여러 쿼리로 스트레스 테스트하는 실험으로 eviction이 가장 취약한 바로 그 실패 모드를 직접 재현해 검증한다.'
    results: '매칭된 20% KV 예산에서 VarRate는 두 모델 모두 비압축 대비 **0.8점** 이내를 유지해 테스트된 것 중 가장 강한 매칭-메모리 압축기다. Query-agnostic reuse 하에서는 query-aware eviction이 11-15점을 잃는 반면 VarRate는 **3.5-5.5점**만 잃는다. Palu를 전 예산에서 확실히 능가하며, KVzip 정확도의 **1점 이내**에 KVzip 대비 **약 1/8** prefill 비용으로 도달한다. Conclusion에 따르면 자체 균일-랭크 ablation 대비 두 모델 가문에서 각각 **2.22점, 17.50점** 개선이며, 사용하는 salience 신호 자체는 오라클 대비 1점 이내다.'
    comparison: '메모리가 매칭된 양자화 대비로는 이기지 못하고 동률이다 — 저자들은 양자화를 이겼다고 주장하지 않고, 랭크와 비트폭이라는 두 축이 상보적이며 실제로 함께 쌓인다(compose)는 것만 실증적으로 확인한다. KVzip과는 4개 설정 중 3개에서 동급, 1개에서 뒤지지만 비용은 약 8분의 1이다.'
    significance: '"학습이 필요 없다"와 "재사용에 강건하다"가 서로 상충한다는, 두 주류 압축 기법이 암묵적으로 전제해온 가정이 사실이 아님을 보여준다. RAG나 멀티턴 에이전트처럼 문맥을 한 번 캐싱해 반복 질의하는 서빙 스택에 특히 유의미하며, "삭제 대신 배분"이라는 원칙이 랭크를 넘어 다른 등급형 충실도 압축 축으로 일반화될 가능성을 제시한다(리뷰어 판단).'
    limitations: '논문 자체에 별도의 Limitations 절은 없다. Conclusion에서 스스로 밝히는 한계는 두 가지다 — 메모리가 매칭된 양자화 대비 이기지 못하고 동률이라는 점, 그리고 query-agnostic reuse 전용으로 설계된 KVzip과 비교했을 때 4개 설정 중 1개에서는 여전히 뒤진다는 점이다. 평가가 두 모델 가문(Llama-3.1-8B, Qwen2.5-7B)에 한정된다는 점은 리뷰어가 덧붙인다(리뷰어 판단).'
    future_work: '논문 내 명시된 future-work 절은 없다. Conclusion은 "값싼 신호를 교체하는 대신 감당 가능하게 만드는 것으로 재사용 강건성을 살 수 있다"는 결론으로 끝난다.'
    resources: '공식 GitHub나 코드 저장소 링크는 논문 본문·HTML 어디에도 없다 — 공개 링크 확인 안 됨. arXiv 초록 페이지만 curl로 200 응답을 확인했다.'
  en:
    background: 'The key-value (KV) cache is the dominant memory bottleneck in long-context LLM inference. Training-free compression splits into two dominant families — token selection / eviction (SnapKV, Ada-KV) scores each token from a query-attention observation window and drops the low scorers, while uniform low-rank coding (Palu-style) keeps every token but projects all of them onto the same reduced-rank subspace. Quantization (KIVI, KVQuant, QJL) is a third, orthogonal axis that lowers numerical precision instead.'
    problem: 'Both dominant families break under query-agnostic reuse — compressing a document once and serving many queries against it (prefix caching, multi-turn dialogue). Eviction''s salience score is essentially "what the current query needs," so when the compression-time query differs from later queries the signal goes stale, and because eviction is irreversible, accuracy collapses by 11-15 points. Uniform low-rank coding never goes stale (it isn''t query-dependent at all) but spends its rank budget uniformly, wasting as much rank on tokens that will never matter as on the ones that will.'
    prior_limits: 'KVzip scores tokens by a query-agnostic reconstruction criterion and is robust to reuse, but obtaining that score requires re-encoding the entire context through the full model several times — several times the prefill cost it is meant to save. Prior attempts at per-token rank variation exist but each pays a different price: DynaKV learns its gate via fine-tuning, while OjaKV is training-free but splits rank by fixed position (anchor tokens vs. the rest) rather than actual query salience.'
    goal: 'Never drop a token, while still spending the rank budget where it actually matters — using only the same cheap query-salience signal eviction methods already compute, with no additional training.'
    method: '**VarRate** allocates rank via water-filling: r_t = clip(r_min + λ·ŝ_t, r_min, R), raising the "water level" λ until the budget B is met, clipping any token that hits the rank cap R and redistributing its surplus (the standard water-filling recursion, Algorithm 1). The salience score ŝ_t is a SnapKV-style query-attention score over a short observation window. A nonzero floor r_min means no token ever drops to rank 0 (full deletion) — a misjudged token is coarsened, never discarded.'
    key_idea: 'The core observation is that both dominant families'' failures come from two independently-chosen design decisions — how you score importance, and how you respond to a low score — that happen to combine badly. VarRate keeps eviction''s cheap scoring but swaps the destructive response (delete) for a reversible one (coarsen) — like a storage-strapped photo app quietly lowering a thumbnail''s resolution instead of deleting the photo. That single swap turns a stale signal from catastrophic into survivable.'
    validation: 'Evaluated on Llama-3.1-8B and Qwen2.5-7B across 16 LongBench tasks. At matched memory budgets, compared against its own uniform-rank ablation, the published low-rank codec Palu, quantization baselines, and KVzip (the method purpose-built for query-agnostic reuse); a dedicated stress test — compress once, then query many times — directly reproduces the exact failure mode eviction is most vulnerable to.'
    results: 'At a matched 20% KV budget, VarRate stays within **0.8 points** of the uncompressed model on both model families, making it the strongest matched-memory compressor tested. Under query-agnostic reuse, query-aware eviction loses 11-15 points while VarRate loses only **3.5-5.5 points**. It strictly dominates Palu at every budget and comes within **~1 point** of KVzip''s accuracy at **~1/8th** KVzip''s prefill cost. Per the conclusion, it beats its own uniform-rank ablation by **2.22 and 17.50 points** on the two model families respectively, and its salience signal lands within a point of an oracle upper bound.'
    comparison: 'Against memory-matched quantization, VarRate ties rather than wins — the authors do not claim to beat quantization, only that rank and bit-width are complementary axes that empirically compose. Against KVzip it is accuracy-equivalent in three of four settings and behind in the fourth, at roughly one-eighth the cost.'
    significance: 'Shows that "training-free" and "robust under reuse" are not actually in tension the way the two dominant compression families implied. Particularly relevant to serving stacks that cache a context once and query it repeatedly (RAG, multi-turn agents), and suggests the "allocate, don''t evict" principle could generalize beyond rank to other graded-fidelity compression axes (reviewer judgment).'
    limitations: 'The paper has no dedicated Limitations section. Two limitations are self-stated in the Conclusion: it ties (does not beat) memory-matched quantization, and it still trails KVzip — a method purpose-built for query-agnostic reuse — in one of four evaluated settings. That evaluation is confined to two model families (Llama-3.1-8B, Qwen2.5-7B) is a reviewer addition (reviewer judgment).'
    future_work: 'Not stated in the paper. The Conclusion ends on the finding itself — reuse robustness can be bought by making a cheap signal survivable rather than by paying to replace it — with no explicit forward-looking directions.'
    resources: 'No official GitHub repository or code release is linked anywhere in the paper or its HTML rendering — no public release verified. Only the arXiv abstract page was confirmed (curl, HTTP 200).'
thread:
  ko: |-
    학습이 필요 없는 KV 압축은 두 갈래로 갈라져 발전해 왔다. 한쪽은 eviction 계보 — StreamingLLM의 위치 기반 선택에서 시작해 SnapKV/Ada-KV가 쿼리 attention 관측 윈도우로 점수를 매기는 방식으로 진화했다. 다른 한쪽은 저랭크 계보 — Palu 같은 방법이 모든 토큰을 유지하되 공유 저차원 부분공간에 균일하게 투영한다. 두 계보는 서로 다른 실패 지점을 갖는다는 것이 오래도록 당연시되어 왔다: eviction은 삭제라는 대가로 정확도를, 저랭크는 균일 지출이라는 대가로 효율을 잃는다는 식이다.

    VarRate의 개념적 이동은 이 두 계보가 사실 하나의 원인을 공유한다는 관찰에서 시작한다 — "무엇이 중요한가"를 채점하는 방식과 "낮은 점수에 어떻게 반응하는가"를 결정하는 방식은 원래 독립적인 두 선택인데, eviction은 이 둘을 (좋은 채점 + 나쁜 반응)으로, 균일 저랭크는 (반응 없음 + 나쁜 채점)으로 묶어왔다는 것이다. VarRate는 eviction의 값싼 쿼리 salience 채점은 그대로 가져오되, 반응만 water-filling 기반의 가변 랭크 배분으로 바꾼다 — 삭제를 거칠게 하기로 대체하는 것. KVzip이 이미 재사용에 강건한 채점 방식을 제시했지만 전체 모델 재인코딩이라는 무거운 대가를 치렀던 것과 달리, VarRate는 기존의 값싼 신호를 그대로 쓰면서 반응 쪽만 고쳐 비슷한 강건성을 훨씬 싸게 얻는다.

    이 논문이 여는 다음 질문은 명확하다 — water-filling으로 배분되는 자원이 랭크일 필요는 없다. 같은 "채점은 그대로, 반응만 가역적으로" 원칙이 비트폭 배분, 혹은 지금까지 이 논문이 별개의 축으로 다뤄온 양자화 자체에도 적용될 수 있을까? 논문은 랭크와 비트폭이 상보적이라 함께 쌓인다는 것만 보이고, 그 이상은 다루지 않는다. Eviction이냐 저랭크냐가 아니라 "무엇을 등급화할 것인가"가 다음 세대 KV 압축 설계의 실제 질문이 될 수 있음을 이 논문은 넌지시 보여준다.
  en: |-
    Training-free KV compression has grown along two separate lineages. One is the eviction line — starting from StreamingLLM's positional selection and evolving through SnapKV/Ada-KV's query-attention-window scoring. The other is the low-rank line — methods like Palu keep every token but project all of them uniformly onto a shared reduced-rank subspace. It has long been taken for granted that these two lineages fail in different ways: eviction trades accuracy for the cost of deletion, uniform low-rank trades efficiency for the cost of spending evenly.

    VarRate's conceptual move starts from noticing these two failures actually share one root cause — scoring "what matters" and deciding "how to respond to a low score" are two independent choices, and eviction bundles them as (good scoring + bad response) while uniform low-rank bundles them as (no response + bad scoring). VarRate keeps eviction's cheap query-salience scoring intact and changes only the response, via water-filling-based variable rank allocation — replacing deletion with coarsening. Where KVzip already offered a reuse-robust scoring method but paid for it with full-model re-encoding, VarRate keeps the existing cheap signal and fixes only the response side, buying similar robustness far more cheaply.

    The question this paper leaves open is clear — there's no reason the water-filled resource has to be rank. Could the same "keep the scoring, make the response reversible" principle apply to bit-width allocation, or to quantization itself, which this paper treats as a separate, merely-composable axis? The paper shows only that rank and bit-width are complementary and stack empirically, going no further. Not eviction-versus-low-rank, but "what should be graded" may be the real design question for the next generation of KV compression.
sparks:
  - ko: '논문이 다루지 않고 남겨둔 질문 — water-filling으로 배분하는 자원이 저랭크가 아니라 비트폭이라면, 같은 "채점은 그대로 두고 반응만 가역적으로 바꾼다"는 원칙이 양자화 자체에도 통할까?'
    en: "A question the paper leaves untouched — if the water-filled resource were bit-width instead of rank, would the same 'keep the scoring, make the response reversible' principle work for quantization itself?"
  - ko: 'KVzip 대비 4개 설정 중 1개에서 여전히 뒤진다는 논문 스스로 밝힌 결과 — 그 한 설정이 무엇을 공유하는지 분석하면 남은 격차의 원인을 좁힐 수 있을지 모른다.'
    en: "The paper's own admission that it still trails KVzip in one of four settings — analyzing what that one setting has in common could narrow down what's driving the remaining gap."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
