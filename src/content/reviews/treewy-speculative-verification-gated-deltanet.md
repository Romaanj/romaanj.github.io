---
title: "TreeWY: Speculative Verification for Gated DeltaNet Hybrids"
arxivId: "2608.20961"
date: "2026-08-25"
tags:
  - speculative-decoding
  - hybrid-architecture
  - serving
topic: serving
summary: TreeWY removes the per-draft-position recurrent-state snapshot that makes wide speculative-decoding draft trees memory-infeasible for hybrid softmax+Gated-DeltaNet models, by rewriting the gated delta rule as a tree-structured WY transform that verifies every draft node with one triangular solve and reconstructs only the accepted state on commit.
summary_ko: TreeWY는 하이브리드(소프트맥스+Gated DeltaNet) 모델에서 넓은 추측 디코딩 드래프트 트리를 메모리 측면에서 불가능하게 만들던 '드래프트 위치별 순환 상태 스냅샷'을 제거한 논문. gated delta rule을 트리 구조 WY 변환으로 다시 써서, 모든 드래프트 노드를 하나의 삼각 연립방정식 풀이로 검증하고 커밋 시 채택된 상태 하나만 복원한다.
links:
  - gated-deltanet
  - dominotree
  - dcut-batched-specdec
resources:
  - label: arXiv abstract
    url: https://arxiv.org/abs/2608.20961
  - label: arXiv PDF
    url: https://arxiv.org/pdf/2608.20961
figures: []
analysis:
  ko:
    background: 최신 오픈 하이브리드 모델은 대부분 층을 소프트맥스 어텐션 대신 선형 어텐션(Gated DeltaNet, GDN)으로 구성한다. GDN 층은 매 시점 하나의 고정 크기 순환 상태 행렬만 유지하므로, 문맥이 길어져도 메모리가 늘지 않는다는 것이 하이브리드 아키텍처의 존재 이유다.
    problem: 이 장점이 추측 디코딩(speculative decoding) 검증 단계에서는 정반대로 뒤집힌다. 드래프트 토큰 k개를 한 번의 병렬 순전파로 검증하는 동안 GDN의 순환 상태는 이미 k번째 토큰 이후까지 진행돼버리는데, 이 상태는 소프트맥스 KV 캐시처럼 포인터만 옮겨 부분적으로 되돌릴 수 없는 손실 압축 표현이다.
    prior_limits: 그래서 현재 시스템(vLLM, SGLang 기본값)은 모든 드래프트 위치마다 전체 순환 상태를 하나씩 스냅샷해 둔다. 문제는 이 스냅샷들이 트리의 여러 분기 사이에서 공유될 수 없다는 것으로, 드래프트 트리가 넓어질수록(즉 노드 수가 많아질수록) 필요한 스냅샷 수가 그대로 늘어나 폭이 넓고 수용률이 높은 트리는 메모리상 실행이 불가능해진다.
    goal: 목표는 GDN 하이브리드 모델에서 드래프트 체인은 물론 폭넓은 드래프트 '트리'까지, 노드별 전체 상태 스냅샷 없이 검증하고 롤백할 수 있는 방법을 찾는 것이다.
    method: "TreeWY는 gated delta rule을 '감쇠 가중 덧셈 어텐션 + 보정된 값(pseudo-value)' 형태로 다시 쓴 뒤, 드래프트 트리의 노드들을 DFS 선위 순서로 배치해 모든 노드의 pseudo-value를 하나의 하삼각 연립방정식으로 한 번에 풀어낸다. 노드 하나가 채택되면, 같은 풀이 결과로부터 그 노드까지의 상태만 다시 조립해 커밋한다 -- 재귀를 순회할 필요도, 노드별 상태를 저장할 필요도 없다."
    key_idea: "핵심 아이디어는 DeltaNet 자체의 체인용 WY 변환을 '트리'로 일반화하는 것이다. 저장 대상을 노드당 전체 상태 행렬에서 작은 pseudo-value 행렬 하나로 바꾸면(d_k=d_v=128 기준 헤드당 128배 더 작은 객체), 체인이든 트리든 저장 블록 수가 N+1개에서 단 1개로 줄어든다. 이는 마치 매 갈림길마다 지도 전체를 복사해두는 대신, 지금까지 걸어온 경로를 하나의 압축된 좌표로만 기록해두고 필요할 때 그 좌표에서 되짚어 계산하는 것과 비슷하다."
    validation: "Qwen3.5-35B-A3B(TP1)와 397B-A17B(TP8) 두 규모의 하이브리드 모델을 vLLM 위에서 B200 GPU로 서빙하며, ShareGPT·spec-bench·BurstGPT를 포함한 6개 워크로드와 GPU 메모리 사용률 3단계, 동시성 여러 단계를 스윕해 검증했다. 정확성은 175개의 짝지어진 (데이터셋, 동시성, 메모리사용률) 지점에서 store-all 기준선과 수용 길이를 직접 비교해 확인했다."
    results: "수용 길이는 기준선과 거의 동일했고(평균 차이 **0.039**, 최대 **0.33**), 동일 부하에서 최대 KV 사용량이 **2~3배** 낮아 훨씬 적게 요청을 대기시킨다(한 조건에서 1365 대 2531건 수용). 메모리가 실제 병목인 지점에서는 스윕상 최대 **1.49배** 처리량, 약 **40배** 낮은 p99 TTFT를 보였고, 병목이 아닌 지점에서는 KV를 2~3배 덜 쓰면서도 처리량은 몇 퍼센트(0.97~0.99배) 낮았다. 드래프트 트리 폭을 넓혀도 저장 블록은 계속 1개로 유지되는 반면(store-all은 (1,1,1)에서 (3,3,3)까지 블록 수가 **10배**(4→40) 증가), 수용 길이는 폭이 넓어질수록 계속 상승했다(1.883→2.786)."
    comparison: "논문은 스스로 두 가지 정직한 비교를 내놓는다. 같은 삼각 연립방정식을 상태 지연 기록(deferred materialization) 방식으로 푸는 동시 진행 연구 ReplaySSM(Dao AI Lab/NVIDIA, vLLM RFC 및 TensorRT-LLM에 탑재 중)과 비교했을 때, 메모리 절감폭은 비슷하지만 순수 처리량에서는 ReplaySSM이 오히려 앞선다(체인 기준 1.12~1.20배 대 TreeWY의 0.99~1.08배) -- 저자들은 이를 스케줄링 차이로 추정하지만 '검증되지 않은 귀속'이라고 명시한다. 또 다른 동시 진행 연구 Bole(SGLang 대상)은 코드가 공개되지 않아 직접 비교하지 못했다고 밝힌다."
    significance: "효율적 AI 관점에서 이 논문은, 순환 상태를 쓰는 선형-어텐션 하이브리드 아키텍처가 확산되면서 새로 생긴 '추측 디코딩 트리 검증'이라는 좁지만 실질적인 병목을 정확히 짚어낸다. 소프트맥스 KV 캐시 세계에서는 당연했던 '트리 분기 간 공유'가 순환 상태에는 없다는 구조적 차이를 드러내고, 이를 트리 구조 선형대수 재구성으로 해결한 사례로서 하이브리드 아키텍처 서빙 스택 설계에 직접적인 참고가 된다."
    limitations: "논문이 직접 밝힌 한계: 트리 모드(분기 폭 1 초과)는 비인과적 조상 마스크가 필요해 CUDA 그래프로 캡처할 수 없고, 그 결과 폭을 넓혀 얻는 수용 길이 이득보다 단계당 비용 증가가 아직 더 크다 -- 즉 트리는 '가능'하지만 아직 처리량 이득은 아니다. ReplaySSM 대비 체인 처리량 열세의 원인(스케줄링 차이 추정)은 저자들 스스로 검증하지 않은 귀속이라고 인정한다. 검증은 Qwen3.5 계열 GDN 하이브리드 한 모델군, 두 규모에서만 수행했다."
    future_work: "논문이 직접 밝힌 다음 단계: (1) 트리 검증 경로를 하나의 CUDA 그래프로 캡처 가능한 커널로 통합하는 것, (2) ReplaySSM·Bole이 시사하는 '상태 기록을 지연시키는' 방식을 직접 구현해 처리량 격차의 원인을 검증하는 것, (3) 두 번째 모델 아키텍처로 확장하는 것."
    resources: 논문 본문에서 vLLM 포크로 구현했다고 밝히지만 구체적인 공개 저장소 링크는 확인되지 않았다. 공개 코드 저장소는 확인 안 됨.
  en:
    background: Recent open hybrid models mix softmax-attention layers with linear-attention Gated DeltaNet (GDN) layers. A GDN layer keeps just one fixed-size recurrent state matrix per head, so its memory footprint stays constant as context grows -- this is the entire reason hybrids are attractive for ordinary decoding.
    problem: That advantage inverts during speculative-decoding verification. Verifying k draft tokens in one parallel forward pass advances the GDN recurrent state all the way past token k before any acceptance decision is made, and unlike a softmax KV cache (which rolls back for free by moving a pointer), this state is a lossy summary that cannot be partially undone.
    prior_limits: Current systems (vLLM, SGLang by default) work around this by snapshotting the full recurrent state at every single draft position. The catch is that these snapshots cannot be shared across branches of a draft tree, so the number of snapshots needed grows directly with tree size -- making wide, high-acceptance draft trees memory-infeasible.
    goal: The goal is a way to verify and roll back GDN hybrid drafts -- chains and wide trees alike -- without storing a full state snapshot per node.
    method: "TreeWY rewrites the gated delta rule as decayed additive attention with a corrected 'pseudo-value,' lays the draft tree's nodes out in DFS pre-order, and solves for every node's pseudo-value at once via a single lower-triangular linear system. Once a node is accepted, its continuation state is reconstructed from that same solve -- no recurrence to walk, no per-node state to keep around."
    key_idea: "The core move is generalizing DeltaNet's own chain-mode WY transform to trees. Swapping a full per-node state matrix for one small pseudo-value matrix (a 128x smaller object per head at d_k=d_v=128) collapses the stored-block count from N+1 down to one, for a chain or a tree alike. It is a bit like replacing a full map-copy at every fork in a path with a single compressed coordinate of the route so far, from which any point can be recomputed on demand."
    validation: "The method was tested serving two scales of a hybrid model, Qwen3.5-35B-A3B (TP1) and 397B-A17B (TP8), in vLLM on B200 GPUs, sweeping six workloads (including ShareGPT, spec-bench, BurstGPT), three GPU-memory-utilization levels, and multiple concurrency settings. Correctness was checked across 175 matched (dataset, concurrency, memory-utilization) points by comparing acceptance length directly against the store-all baseline."
    results: "Acceptance length tracked the baseline almost exactly (mean difference **0.039**, max **0.33**), and peak KV usage was **2-3x lower** at matched load, admitting far more requests before preempting (1365 vs. 2531 in one setting). Where memory was the actual constraint, TreeWY reached up to **1.49x** throughput and roughly **40x** lower p99 time-to-first-token in the sweep; where it wasn't, throughput trailed by a few percent (0.97-0.99x) despite still using 2-3x less KV. Widening the draft tree kept the stored-block count flat at one, while a store-all baseline's cost grew **10x** (4 to 40 blocks) over the same range, and acceptance length kept climbing with tree width (1.883 to 2.786)."
    comparison: "The paper offers two honest concurrent-work comparisons. Against ReplaySSM (Dao AI Lab/NVIDIA, shipping in a vLLM RFC and TensorRT-LLM), which solves the same triangular system but defers when the state is materialized, TreeWY frees comparable memory but actually loses on raw chain throughput (ReplaySSM: 1.12-1.20x its own baseline vs. TreeWY's 0.99-1.08x) -- attributed to scheduling differences, which the authors explicitly flag as an unvalidated attribution rather than a tested claim. A second concurrent method, Bole (targeting SGLang), could not be benchmarked head-to-head because no code was available."
    significance: "From an efficient-AI lens, this paper names and fixes a specific, previously under-examined bottleneck created by the rise of recurrent-state hybrid architectures: the branch-sharing that speculative-decoding trees take for granted with a softmax KV cache simply does not exist for a recurrent state, and that gap only shows up once you try to verify wide draft trees rather than chains. Solving it with a closed-form tree-structured linear-algebra rewrite is a concrete, directly reusable pattern for anyone building a serving stack around recurrent-state hybrids."
    limitations: "Paper-stated: tree mode (branching factor above one) needs a non-causal ancestor mask that cannot be captured into a CUDA graph, so the per-step cost of a wider tree currently outweighs the acceptance-length gain it buys -- trees are enabled and correct, not yet a throughput win. The cause of TreeWY's chain-throughput gap versus ReplaySSM (attributed to scheduling) is explicitly acknowledged as untested. Evaluation covers a single model family (Qwen3.5 GDN hybrids) at two scales."
    future_work: "The paper names three concrete next steps: (1) fusing the tree-verification path into a single CUDA-graph-capturable kernel, (2) implementing the deferred-state-write idea that both ReplaySSM and Bole point to, in order to actually test whether it closes the throughput gap, and (3) extending the approach to a second model architecture."
    resources: The paper states the method is implemented as a vLLM fork, but no public repository link could be confirmed from the text. No public code release verified.
thread:
  ko: |-
    선형-어텐션/순환-상태 하이브리드 아키텍처(Gated DeltaNet 등)는 문맥이 길어져도 메모리가 늘지 않는다는 장점 덕에 최근 오픈 모델의 표준 구성 요소가 됐다. 동시에 추측 디코딩은 트리 형태 드래프트(Medusa류)로 진화하며 한 번에 여러 후보를 검증해 처리량을 높여 왔다. 이 두 흐름은 서로 독립적으로 발전해 왔고, TreeWY 이전까지는 순환 상태 하이브리드에서 넓은 드래프트 트리를 검증하는 문제 자체가 제대로 다뤄지지 않았다.

    TreeWY가 만드는 전환은, '순환 상태는 KV 캐시처럼 부분적으로 되돌릴 수 없다'는 제약을 스냅샷을 늘리는 방식이 아니라 아예 다른 계산 형태(트리 구조 닫힌 형식 선형대수)로 우회하는 것이다. DeltaNet 자체가 이미 갖고 있던 체인용 WY 변환을 트리로 확장함으로써, 노드마다 상태를 저장하는 대신 하나의 작은 보정값 행렬만으로 트리 전체를 검증하고 필요한 상태만 사후 복원한다.

    논문이 스스로 인정하듯 이 방향은 아직 완성형이 아니다 -- 트리 검증은 '가능'해졌지만 CUDA 그래프 캡처가 안 돼 아직 처리량 이득으로 이어지지 않고, 비슷한 시기에 나온 ReplaySSM·Bole 같은 경쟁 접근과 견주어도 어느 설계가 최종적으로 이길지는 정해지지 않았다. 다음 단계는 이 닫힌 형식을 실제로 빠른 커널로 만드는 엔지니어링, 그리고 '상태를 언제 기록할 것인가'라는 설계축을 정면으로 실험하는 것이다.
  en: |-
    Recurrent-state hybrid architectures (Gated DeltaNet and similar) have become a standard building block in recent open models because their memory footprint stays flat as context grows. Separately, speculative decoding has evolved toward tree-shaped drafts (the Medusa line) that verify several candidates per step to push throughput higher. These two lines developed independently, and before TreeWY, verifying a wide draft tree against a recurrent-state hybrid was not really addressed.

    The shift TreeWY makes is refusing to solve "recurrent state can't be partially rolled back like a KV cache" by adding more snapshots, and instead changing the computation itself into a closed-form, tree-structured linear-algebra problem. By generalizing DeltaNet's own chain-mode WY transform to trees, it verifies the whole draft tree with one small correction matrix instead of one state snapshot per node, reconstructing only the state that is actually needed afterward.

    As the paper itself is candid about, this is not a finished story -- tree verification is now possible but not yet a throughput win because it cannot be captured into a CUDA graph, and it is not yet clear which of several concurrent designs (TreeWY, ReplaySSM, Bole) will win out. The next steps are turning the closed form into an actually fast, graph-capturable kernel, and directly testing the "when do you write the state" design axis that both concurrent methods point to.
sparks:
  - ko: 논문은 트리 검증 경로가 CUDA 그래프로 캡처되지 않아 폭을 넓혀도 아직 처리량 이득이 나지 않는다고 밝힌다 -- 이 커널을 그래프 캡처 가능한 형태로 재구성하면 드래프트 폭 확장이 실제 처리량 이득으로 이어지는 지점이 어디인지가 열린 질문이다.
    en: The paper states that the tree-verification path cannot be captured into a CUDA graph, which is why wider trees are not yet a throughput win -- restructuring that kernel to be graph-capturable would open the question of exactly where wider drafts start paying off in real throughput.
  - ko: 저자들은 ReplaySSM 대비 체인 처리량 격차를 '지연 상태 기록' 스케줄링 차이로 추정하지만 직접 검증하지는 않았다고 밝힌다 -- 동일한 닫힌 형식 위에서 상태 기록 시점만 바꿔가며 실험하면 이 격차의 진짜 원인을 가릴 수 있을 것이다.
    en: The authors attribute the chain-throughput gap versus ReplaySSM to deferred-state-write scheduling but say they have not validated this -- testing the same closed form under different state-write timings would isolate whether that is really the source of the gap.
source: autosweep
---

## Notes

<!-- Structured 13-item bilingual analysis lives in the frontmatter above. -->
