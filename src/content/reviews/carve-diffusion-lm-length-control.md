---
title: "CARVE: Counterfactual-Aware Reveal with Verified Expansion for Diffusion Language Models"
arxivId: "2608.30922"
date: 2026-09-02
tags: ["diffusion-llm", "dllm-acceleration", "decoding"]
topic: 'diffusion-llm'
summary: "CARVE lets a masked diffusion language model grow its answer canvas mid-generation, accepting the extra space only when a counterfactual stability check confirms the insertion will not disturb the model's existing predictions."
summary_ko: 'CARVE는 마스크 확산 언어모델이 생성 도중 답변 캔버스를 늘리도록 하되, 그 확장이 기존 예측을 흔들지 않는다는 반사실적 안정성 검증을 통과했을 때만 받아들인다.'
links: ["survival-guided-length-control", "llada", "fast-dllm", "serving-masked-dllm-hardware"]
resources:
  - label: "arXiv abstract"
    url: "https://arxiv.org/abs/2608.30922"
  - label: "arXiv PDF"
    url: "https://arxiv.org/pdf/2608.30922"
figures:
  - src: "/figures/carve-diffusion-lm-length-control/fig1.png"
    caption: "CARVE's verified-expansion loop: a candidate canvas expansion is scored by Jensen-Shannon divergence against the original canvas and only committed if the divergence stays below threshold."
    caption_ko: "CARVE의 검증된 확장 루프: 후보 캔버스 확장을 원본 캔버스와의 Jensen-Shannon 발산으로 채점하고, 발산이 임계값 이하일 때만 확장을 확정한다."
    credit: "Figure 1 from arXiv:2608.30922 — authors' figure"
analysis:
  ko:
    background: '마스크 확산 언어모델(dLLM)은 응답을 미리 정한 길이의 "캔버스" 위에서 양방향으로 병렬 디노이징한다. **LLaDA**, **Dream** 같은 모델이 이 방식으로 코드·수학 추론에서 경쟁력 있는 성능을 보여왔다.'
    problem: '문제는 캔버스 길이를 디코딩 시작 전에 고정해야 한다는 것이다. 너무 짧으면 추론이나 코드가 중간에 잘리고, 너무 길면 불필요한 마스크 위치에 연산을 낭비하며 오히려 품질을 해칠 수 있다. 적정 길이는 프롬프트마다 다르다.'
    prior_limits: '선행 연구 DAEDAL은 모델의 confidence·EOS 회피 신호를 보고 캔버스를 늘릴지 판단하는 학습-불필요 방법을 제안했다. 하지만 확장 자체가 디노이징 문제를 바꾼다는 두 번째 질문은 다루지 않았다 — 새 마스크를 끼워 넣으면 이미 풀리지 않은 다른 위치의 예측이 흔들릴 수 있다.'
    goal: '캔버스를 늘릴지 말지를 confidence 휴리스틱이 아니라, 그 삽입이 기존 예측을 실제로 안정적으로 유지하는지에 대한 **검증된 판단**으로 바꾸는 것이 목표다.'
    method: 'CARVE(Counterfactual-Aware Reveal with Verified Expansion)는 특정 리빌 단계 직전에 추가 마스크 토큰 삽입을 제안한다. 확장된 캔버스에서 한 번 순전파를 돌려, 원본과 확장본에서 공통으로 존재하는 미해결 위치들의 예측 분포를 비교한다. 두 분포 간 평균 Jensen-Shannon(JS) 발산이 고정 임계값 아래면 확장을 확정하고, 아니면 폐기하고 원본 캔버스로 계속한다.'
    key_idea: '핵심 발상은 길이 확장을 "이 삽입이 있었다면 모델이 남은 위치들에 대해 거의 같은 예측을 했을까?"라는 **반사실적 질문**으로 바꾸는 것이다. 사진의 클리핑된 하이라이트를 복원할지 판단하는 것과 비슷하게, 새 공간을 추가해도 기존 노출 값이 크게 안 변해야 안전한 확장으로 간주한다. 이 방식은 full-canvas 디코딩(Dream)과 블록 단위 디코딩(LLaDA)에 동일하게 적용된다.'
    validation: 'HumanEval, MBPP, MATH-500, GSM8K 네 벤치마크에서 Dream-v0-Instruct-7B, LLaDA-1.5, LLaDA-8B-Instruct 세 모델에 대해, 고정 길이 베이스라인 및 가장 가까운 학습-불필요 경쟁자 DAEDAL과 비교했다.'
    results: '세 모델 모두에서 평균 정확도가 고정 길이 베이스라인 대비 소폭 상승했다 (Dream **+0.92pp**, LLaDA-1.5 **+1.03pp**, LLaDA-8B **+0.49pp**). 동시에 forward-pass FLOPs는 일부 세팅에서 베이스라인의 **약 절반**까지 줄었다 — 확장으로 인한 추가 순전파 비용을 적응적 리빌과 EOS 크로핑이 상쇄하기 때문이다.'
    comparison: 'DAEDAL과 비교하면 CARVE가 세 모델 평균 모두에서 더 낫거나 같았다. 다만 태스크별로는 혼재된 결과다 — 예컨대 Dream의 MBPP는 베이스라인 대비 -0.20pp로 소폭 하락했고, DAEDAL은 다른 태스크에서 더 큰 폭(GSM8K -3.49pp)으로 손해를 봤다.'
    significance: 'KV 캐시나 양자화가 아니라 **캔버스 길이 자체를 검증 가능한 방식으로 바꾸는 것**도 dLLM 추론 비용을 줄이는 독립적인 축이라는 점을 보여준다. 특히 block-diffusion 디코딩까지 같은 알고리즘이 통한다는 점은 실제 서빙 파이프라인에 끼워 넣기 쉬운 형태라는 뜻이다.'
    limitations: '(저자 명시) 삽입 크기는 현재 고정값(k=16)이며 프롬프트별 불확실성에 적응하지 않는다. (저자 명시) JS 발산 정렬은 삽입 전후 공통 위치만 비교하고 새로 추가된 마스크 위치는 검증에서 제외한다 — 저자 스스로 "단순하지만 다소 경직적일 수 있다"고 인정한다. (리뷰어 판단) 평균 개선폭이 1.5pp를 넘지 않고 태스크별 하락도 같은 표에 함께 나타나므로, "확실히 더 낫다"보다는 "손해는 안 보고 때때로 돕는, 근거가 명확한 방법"으로 읽는 것이 정확하다.'
    future_work: '(저자 명시) 삽입 크기를 프롬프트별 불확실성에 맞춰 적응시키는 것과, 새로 삽입된 위치까지 포함하는 더 유연한 정렬·발산 기준을 향후 과제로 제시한다.'
    resources: '코드가 GitHub(wailji/CARVE)에 공개되어 있다고 논문에 명시되어 있으나, 리뷰 시점에 저장소 URL이 접속되지 않아 링크는 확인된 arXiv 초록·PDF만 포함했다.'
  en:
    background: 'Masked diffusion language models (dLLMs) denoise a response bidirectionally and in parallel over a "canvas" of predetermined length. Models like **LLaDA** and **Dream** have shown this approach is competitive on code and math reasoning.'
    problem: 'The catch is that canvas length must be fixed before decoding starts. Too short truncates reasoning or code; too long wastes computation on masked positions that may not be needed and can even hurt quality. The right length is also instance-dependent across prompts.'
    prior_limits: 'Prior work DAEDAL proposed a training-free way to decide when to grow the canvas using the model''s own confidence and EOS-avoidance signals. But it left a second question unaddressed: expansion itself changes the denoising problem — inserting new masks can perturb predictions at other still-unresolved positions.'
    goal: 'The goal is to turn the decision of whether to expand the canvas from a confidence heuristic into a **verified judgment** of whether the insertion actually keeps existing predictions stable.'
    method: 'CARVE (Counterfactual-Aware Reveal with Verified Expansion) proposes inserting additional masked tokens before certain reveal steps. It runs the model once on the candidate expanded canvas and compares the predictive distributions at positions that exist in both the original and expanded canvas. The expansion is committed only if the mean Jensen-Shannon (JS) divergence between the two distributions stays below a fixed threshold; otherwise it is discarded and decoding continues on the original canvas.'
    key_idea: 'The core idea is to reframe length growth as a **counterfactual question**: would the model make essentially the same predictions for the remaining positions if the extra space had been there? Much like judging whether recovering a photo''s clipped highlights would change the rest of the image, an expansion is deemed safe only if it leaves the already-exposed predictions largely unchanged. The same criterion applies unchanged across full-canvas decoding (Dream) and blockwise decoding (LLaDA).'
    validation: 'Evaluated on HumanEval, MBPP, MATH-500, and GSM8K across three instruction-tuned dLLMs — Dream-v0-Instruct-7B, LLaDA-1.5, LLaDA-8B-Instruct — against a fixed-length baseline and DAEDAL, the closest training-free variable-length competitor.'
    results: 'Average accuracy improved modestly over the fixed-length baseline across all three models (Dream **+0.92pp**, LLaDA-1.5 **+1.03pp**, LLaDA-8B **+0.49pp**). At the same time, forward-pass FLOPs dropped to roughly **half** the baseline in some settings, since adaptive reveal and EOS cropping offset the extra forward pass each expansion costs.'
    comparison: 'CARVE beats or ties DAEDAL on the per-model average in all three cases. Per-task results are mixed, though: Dream''s MBPP score dips -0.20pp versus baseline, while DAEDAL loses more elsewhere (e.g. -3.49pp on GSM8K).'
    significance: 'It shows that reshaping **canvas length itself, under a verifiable stability criterion**, is an independent lever for cutting dLLM inference cost — separate from KV-cache or quantization work. That the same algorithm works for blockwise decoding matters for real serving pipelines, since blockwise (SDAR/LLaDA-style) decoding is what current deployed dLLMs actually use.'
    limitations: '(paper-stated) Insertion size is fixed (k=16) and does not adapt to per-prompt uncertainty. (paper-stated) The JS-divergence alignment only compares positions common to both canvases and excludes newly inserted positions from verification — the authors themselves call this "simple... but may be overly rigid." (reviewer judgment) With average gains under 1.5pp and visible per-task regressions in the same tables, the honest read is "does not hurt and sometimes helps, with a genuine mechanism" rather than "clearly better."'
    future_work: '(paper-stated) The authors flag adaptive insertion sizing (tied to per-prompt uncertainty) and softer alignment/divergence criteria that also account for newly inserted positions as direct next steps.'
    resources: 'The paper states code is available on GitHub (wailji/CARVE), but the repository URL did not resolve at review time, so only the verified arXiv abstract and PDF are listed here.'
thread:
  ko: |-
    dLLM의 가변 길이 디코딩 문제는 DAEDAL이 먼저 열었다. "언제 캔버스를 늘릴지"를 모델의 confidence·EOS 신호로 판단하는 학습-불필요 방법이었고, 고정 길이라는 오래된 가정을 흔든 첫 시도였다.

    CARVE가 이 흐름에서 바꾸는 지점은 "늘릴지"가 아니라 "늘렸을 때 안전한지"를 묻는 것이다. 확신도가 낮다고 무조건 캔버스를 늘리는 대신, 실제로 삽입이 이미 풀리지 않은 다른 위치의 예측을 흔드는지를 반사실적으로 검증한다. 양방향 디노이징이라는 dLLM 고유의 성질 — 한 위치의 변화가 다른 위치의 예측에 영향을 준다 — 을 직접 문제 삼은 첫 사례에 가깝다.

    다음으로 열리는 질문은 이 검증 기준을 더 정교하게 만드는 것이다. 지금은 고정 임계값과 고정 삽입 크기를 쓰지만, 프롬프트별 불확실성에 맞춰 삽입 크기 자체를 적응시키거나, 새로 삽입된 위치까지 포함하는 더 유연한 정렬 기준으로 확장될 여지가 크다. 또한 이런 캔버스 구조 변경이 KV 캐시 재사용이나 캐싱 스케줄과 어떻게 상호작용할지도 아직 다뤄지지 않은 영역이다.
  en: |-
    The variable-length decoding problem in dLLMs was first opened by DAEDAL, a training-free method that decides "when to grow the canvas" using the model's own confidence and EOS-avoidance signals — the first real challenge to the long-standing fixed-length assumption.

    CARVE shifts the question from "should we grow it" to "is growing it safe." Instead of expanding whenever confidence looks low, it counterfactually verifies whether the insertion actually destabilizes predictions at other still-unresolved positions. It is one of the first methods to directly confront a property specific to dLLMs — bidirectional denoising, where a change at one position can ripple into predictions elsewhere.

    What it opens next is refining this verification criterion further. The current version uses a fixed threshold and fixed insertion size, but there is clear room to make insertion size itself adaptive to per-prompt uncertainty, or to use a more flexible alignment criterion that also accounts for newly inserted positions. How this kind of mid-decode canvas restructuring interacts with KV-cache reuse or caching schedules is also still an open area.
sparks:
  - ko: "삽입 크기를 프롬프트별 불확실성에 맞춰 적응시키면 — 예를 들어 JS 발산이 임계값에 가까울수록 더 작게 삽입하면 — 평균 이득이 지금보다 커질까?"
    en: "If insertion size adapted to per-prompt uncertainty — say, smaller insertions when JS divergence sits close to the threshold — would the average gain grow beyond what a fixed k=16 achieves?"
  - ko: "새로 삽입된 마스크 위치까지 포함하는 더 유연한 정렬·발산 기준을 쓰면 지금의 '경직적' 검증이 놓치는 안전한 확장을 더 찾아낼 수 있을까?"
    en: "Would a softer alignment/divergence criterion that also covers the newly inserted positions catch safe expansions that the current, admittedly rigid, verification rule misses?"
source: "autosweep"
---

## Notes
