---
title: "MXSens: Sensitivity-Aware Mixed-Precision Quantization for Efficient LLM Inference"
arxivId: "2607.17733"
authors: "Simla Burcu Harma, Danila Mishin, Zhengyuan Su, Ayan Chakraborty, Elizaveta Kostenok, Dongho Ha"
date: 2026-07-21
tags: ["quantization", "compression", "mixed-precision"]
topic: 'compression'
summary: "MXSens allocates 4/6/8-bit mantissa widths per column and layer based on Hessian-guided sensitivity, staying inside hardware-native MXINT microscaling formats (unlike rotation-based methods, which are incompatible with them), and reaches perplexities of 3.77 and 7.63 on LLaMA-2-70B and LLaMA-3-8B under W4A4KV4."
summary_ko: "MXSens는 Hessian 기반 민감도 분석으로 열(column)·층(layer)별로 4/6/8비트 mantissa 폭을 배분하면서 하드웨어 네이티브 MXINT 마이크로스케일링 포맷과의 호환성을 유지하고(회전 기반 기법은 이 포맷과 호환되지 않음), W4A4KV4 설정에서 LLaMA-2-70B/LLaMA-3-8B에 각각 perplexity 3.77/7.63을 달성한다."
links: ["kvquant", "liftquant"]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.17733' }
figures:
  - src: /figures/mxsens/fig1.png
    caption: "MXINT4 error propagation in the q-proj layer under single-layer quantization on LLaMA-2-7B, motivating why quantization sensitivity must be tracked per column and per layer rather than assumed uniform."
    caption_ko: "LLaMA-2-7B의 q-proj 레이어를 단일 층 양자화했을 때 MXINT4 오차 전파 — 양자화 민감도가 균일하다고 가정할 게 아니라 열·층 단위로 추적해야 하는 이유를 보여준다."
    credit: "Figure 1 from arXiv:2607.17733 — authors' figure"
  - src: /figures/mxsens/fig2.png
    caption: "Perplexity dynamics of different bitwidth-allocation strategies on Llama-3-8B, comparing uniform allocation against MXSens's sensitivity-guided 4/6/8-bit assignment."
    caption_ko: "Llama-3-8B에서 서로 다른 비트폭 배분 전략의 perplexity 변화 — 균일 배분과 MXSens의 민감도 기반 4/6/8비트 배분 비교."
    credit: "Figure 2 from arXiv:2607.17733 — authors' figure"
analysis:
  ko:
    background: '4비트 양자화는 효율적인 LLM 추론을 가능케 하지만, 이상치(outlier) 때문에 정확도가 크게 떨어진다. 이를 해결하려는 기존 방법들은 데이터 회전(rotation)이나 혼합-정수 양자화를 쓰는데, 이들은 대개 소프트웨어가 관리하는 스케일링과 잦은 역양자화(dequantization)를 필요로 해 상당한 오버헤드를 낳는다.'
    problem: 'MXINT 같은 마이크로스케일링 포맷은 스케일을 하드웨어에 직접 인코딩해 이런 비효율을 없애지만, 회전 기반 기법과는 근본적으로 호환되지 않는다. 문제는 회전 없이도, 그리고 하드웨어 네이티브 포맷을 유지하면서도 이상치로 인한 정확도 손실을 얼마나 줄일 수 있는가이다.'
    prior_limits: '균일한 비트폭을 전 층·전 열에 걸쳐 적용하는 방식은 단순하지만, LLM 전반에 걸쳐 관측되는 양자화 민감도의 큰 편차를 무시한다. 이상치는 드물지만 극단적인 것부터 흔하지만 완만한 것까지 심각도가 제각각이며, 민감도는 층과 열에 따라 고르지 않게 분포한다는 것이 이 논문의 분석이 밝히는 지점이다.'
    goal: '회전이나 잦은 역양자화 없이, MXINT의 블록 단위 구조를 그대로 활용하면서 세밀한(열·층 단위) 민감도-유도 비트폭 배분으로 정확도-효율 트레이드오프를 개선하는 것이 목표다.'
    method: '**MXSens**는 열(column) 단위·층(layer) 단위 민감도에 따라 4/6/8비트의 혼합 mantissa 폭을 배정하는 학습-불필요 방법이다. 사진의 배경은 압축적으로 저장하고 얼굴처럼 중요한 부분에는 더 많은 화소(비트)를 남겨두는 것과 비슷하다 — 모든 곳에 같은 해상도를 쓰는 대신, 정확도에 가장 큰 영향을 주는 곳에만 더 높은 정밀도를 쓴다.'
    key_idea: '핵심은 이상치의 심각도가 극단적인 것부터 완만한 것까지 다양하고 양자화 민감도가 층·열에 따라 고르지 않다는 관찰을, 회전이라는 무거운 수단이 아니라 MXINT가 이미 갖고 있는 블록 단위 구조 위에서 세밀한 비트 배분으로 직접 해결한다는 점이다. 회전 기반 기법을 포기하는 대신 하드웨어 호환성을 지키면서 같은 문제(이상치)를 다른 축(정밀도 배분)으로 공략한다.'
    validation: 'LLaMA-2-70B, LLaMA-3-8B 등 여러 모델과 태스크(WikiText-2 perplexity, Common Sense QA 0-shot 정확도)에서 평가하며, A4W4KV4·A4W4KV16 두 양자화 스킴에 걸쳐 최신 양자화 방법들과 비교한다. 보정(calibration) 데이터셋 선택(WikiText-2 vs C4), 다양한 평균 비트폭 설정에 대한 ablation도 포함한다.'
    results: 'W4A4KV4 설정에서 MXSens는 LLaMA-2-70B와 LLaMA-3-8B에서 각각 perplexity **3.77**과 **7.63**을 달성해 WikiText-2 상에서 기존 베이스라인 대비 상당히 개선된 결과를 보인다. 논문은 이를 "정확도와 자원 효율 사이의 새로운 균형"이라 표현한다.'
    comparison: '같은 혼합-정밀도 원칙을 쓰는 이전 방법들과 달리, MXSens는 회전 기반 기법이 필요로 하는 소프트웨어 관리 스케일링이나 잦은 역양자화 없이 MXINT라는 하드웨어 네이티브 블록 포맷 안에서 그대로 작동한다 — 회전 기법과 하드웨어 마이크로스케일링 포맷 사이의 근본적 비호환성을 우회하는 것이 이 논문의 실질적 차별점이다.'
    significance: '효율적 LLM 추론에서 "회전이냐 마이크로스케일링 하드웨어 지원이냐"를 양자택일로 여기지 않고 후자를 유지한 채 정확도 손실을 줄일 수 있음을 보여준다. MXINT 같은 포맷이 실제 가속기에 채택되는 상황에서, 회전 기법과의 비호환성 때문에 정확도를 포기할 필요가 없다는 실용적 함의를 갖는다.'
    limitations: '논문 자체의 결론부에는 별도 Limitations 절이 없다(대신 재현성 진술이 있다). 평가가 LLaMA·Mistral·Qwen 계열의 밀집(dense) 트랜스포머에 집중되어 있어, MoE 아키텍처의 전문가(expert) 가중치처럼 민감도 분포가 라우팅 빈도에 따라 크게 달라질 수 있는 구조에도 같은 열·층 단위 민감도 분석이 그대로 적용될지는 논문이 다루지 않는다는 점은 리뷰어가 덧붙인다(리뷰어 판단).'
    future_work: '논문 내 명시된 별도 future-work 절은 없다. 대신 Section 8(재현성 진술)에서 모델·데이터셋·평가 지표, 민감도 계산·비트폭 배분·MXINT 양자화 루틴의 구현 세부사항, 열/층별 민감도 추출과 삼중 양자화 알고리즘의 의사코드 위치를 상세히 명시한다.'
    resources: '논문 본문에 "코드 저장소가 https://github.com/parsa-epfl/mxsens 에 공개되어 있다"고 명시되어 있지만, 확인 시점(curl) 기준 해당 저장소는 404를 반환해 접근되지 않는다 — 공개 링크 확인 안 됨(저자들이 코드를 아직 공개하지 않았거나 저장소가 비공개 상태일 수 있다).'
  en:
    background: '4-bit quantization enables efficient LLM inference, but suffers significant accuracy degradation from outliers. Existing fixes rely on data rotation or mixed-integer quantization, both of which typically require software-managed scaling and frequent dequantization, incurring substantial overhead.'
    problem: 'Microscaling formats such as MXINT eliminate that inefficiency by encoding scales directly in hardware, but they are fundamentally incompatible with rotation-based methods. The problem is how much of the outlier-driven accuracy loss can be recovered without rotation, while staying inside a hardware-native format.'
    prior_limits: 'Applying a uniform bitwidth across every layer and column is simple but ignores the substantial variation in quantization sensitivity observed across LLMs. This paper''s own analysis shows outliers range in severity from rare and extreme to frequent and mild, and that sensitivity is unevenly distributed across layers and columns.'
    goal: 'Without rotation or frequent dequantization, and while keeping MXINT''s block-wise structure intact, improve the accuracy-efficiency trade-off through fine-grained (column- and layer-wise) sensitivity-guided bitwidth allocation.'
    method: '**MXSens** is a training-free method that assigns mixed mantissa bitwidths (4/6/8) based on column- and layer-wise sensitivity. It is similar to compressing the background of a photo aggressively while leaving more pixels (bits) for a face that matters — rather than using the same resolution everywhere, higher precision goes only where it most affects accuracy.'
    key_idea: 'The core move is to address the observation that outlier severity ranges from extreme to mild and that sensitivity is unevenly distributed across layers and columns directly through fine-grained bit allocation on top of the block-wise structure MXINT already has, rather than through the heavier tool of rotation. Instead of abandoning rotation-based methods and losing hardware compatibility, it attacks the same problem (outliers) along a different axis (precision allocation).'
    validation: 'Evaluated across several models (LLaMA-2-70B, LLaMA-3-8B, among others) and tasks (WikiText-2 perplexity, zero-shot Common Sense QA accuracy), compared against state-of-the-art quantization methods across two quantization schemes, A4W4KV4 and A4W4KV16. Ablations cover calibration-dataset choice (WikiText-2 vs. C4) and different average-bitwidth configurations.'
    results: 'Under the W4A4KV4 setting, MXSens achieves perplexities of **3.77** and **7.63** on LLaMA-2-70B and LLaMA-3-8B respectively, substantially improving over existing baselines on WikiText-2. The paper frames this as establishing "a new balance between accuracy and resource efficiency."'
    comparison: 'Unlike prior methods that share the same mixed-precision principle, MXSens operates entirely within the hardware-native MXINT block format, without the software-managed scaling or frequent dequantization rotation-based methods require — sidestepping the fundamental incompatibility between rotation and hardware microscaling formats is this paper''s practical point of differentiation.'
    significance: 'Shows that "rotation vs. hardware microscaling support" need not be an either/or choice for efficient LLM inference — accuracy loss can be reduced while keeping the latter. Practically relevant as formats like MXINT get adopted in real accelerators, meaning accuracy need not be sacrificed for incompatibility with rotation-based techniques.'
    limitations: 'The paper has no dedicated Limitations section in its conclusion (it has a reproducibility statement instead). That the evaluation focuses on dense transformers in the LLaMA/Mistral/Qwen families, and that it''s unclear whether the same column- and layer-wise sensitivity analysis would transfer as-is to MoE architectures, where expert-weight sensitivity could vary drastically with routing frequency, is a reviewer addition (reviewer judgment).'
    future_work: 'Not stated in the paper. Instead, Section 8 (Reproducibility Statement) details the models, datasets, and evaluation metrics used, the implementation details of sensitivity computation, bitwidth allocation, and MXINT quantization routines, and points to the pseudocode locations for column-wise and layer-wise sensitivity extraction and the triplet quantization algorithm.'
    resources: 'The paper states in its text that "our code repository is publicly available at https://github.com/parsa-epfl/mxsens," but as of verification (curl) that repository returns a 404 and does not resolve — no public release verified (the authors may not have published the code yet, or the repository may be private).'
thread:
  ko: |-
    4비트 LLM 양자화에서 이상치 문제를 다루는 계보는 크게 두 갈래로 갈려 왔다. 하나는 회전(rotation) 계보 — QUIK 같은 방법들이 이상치를 통계적으로 흩뿌려 균일 양자화를 더 쉽게 만든다. 다른 하나는 혼합-정밀도 계보 — 민감한 부분에 더 높은 정밀도를 배정한다. 두 계보 모두 정확도를 개선해왔지만, 회전은 소프트웨어 관리 스케일링과 잦은 역양자화라는 대가를 치렀고, 그 대가는 하드웨어가 스케일을 직접 인코딩하는 마이크로스케일링 포맷(MXINT 등)이 등장하면서 근본적인 비호환성으로 굳어졌다.

    MXSens의 개념적 전환은 "회전이냐 마이크로스케일링 하드웨어 지원이냐"를 양자택일로 보지 않고, 혼합-정밀도 계보를 정교화해 회전 없이도 이상치 문제를 공략하는 것이다. 열·층 단위로 민감도를 추출해 4/6/8비트를 배분하는 것 자체는 새로운 발상이 아니지만, 이를 MXINT의 블록 단위 구조와 정확히 맞물리게 설계함으로써 하드웨어 호환성을 희생하지 않는다. 즉 문제를 "이상치를 어떻게 없앨까"에서 "이상치가 있는 곳에만 얼마나 정밀도를 더 줄까"로 재정의하되, 그 재정의를 하드웨어가 이미 지원하는 포맷 안에서 수행한다.

    이 논문이 여는 다음 질문은 이 원칙이 밀집 트랜스포머를 넘어 어디까지 가는가이다. MoE 아키텍처의 전문가별 민감도, 혹은 MXINT 이후 등장할 다른 블록-스케일 포맷(NVFP4 등)에도 같은 민감도-유도 배분 원칙이 그대로 이식될지는 이 논문이 다루지 않는다.
  en: |-
    In 4-bit LLM quantization, tackling the outlier problem has followed two broad lineages. One is the rotation lineage — methods like QUIK statistically spread outliers around to make uniform quantization easier. The other is the mixed-precision lineage — assigning higher precision to sensitive components. Both lineages improved accuracy, but rotation paid for it with software-managed scaling and frequent dequantization, and that cost hardened into a fundamental incompatibility once hardware-native microscaling formats (like MXINT) emerged that encode scales directly in hardware.

    MXSens's conceptual shift is to stop treating "rotation vs. hardware microscaling support" as an either/or choice, and instead refine the mixed-precision lineage to attack the outlier problem without rotation at all. Extracting column- and layer-wise sensitivity to allocate 4/6/8 bits isn't itself a new idea, but designing it to mesh precisely with MXINT's block-wise structure is what avoids sacrificing hardware compatibility. It reframes the problem from "how do we get rid of outliers" to "how much extra precision do we give only where outliers actually are" — while performing that reframing entirely inside a format hardware already supports.

    The question this paper leaves open is how far this principle travels beyond dense transformers. Whether the same sensitivity-guided allocation principle ports cleanly to per-expert sensitivity in MoE architectures, or to other block-scaled formats that may emerge after MXINT (such as NVFP4), is not explored here.
sparks:
  - ko: '평가는 LLaMA·Mistral·Qwen 같은 밀집 트랜스포머에 집중된다 — 라우팅 빈도에 따라 전문가별 민감도가 크게 달라질 수 있는 MoE 아키텍처의 전문가 가중치에도 같은 열·층 단위 Hessian 기반 민감도 분석이 그대로 통할지는 열려 있다.'
    en: "The evaluation focuses on dense transformers like LLaMA, Mistral, and Qwen — whether the same column- and layer-wise Hessian-guided sensitivity analysis transfers as-is to MoE expert weights, where sensitivity could vary drastically with routing frequency, remains open."
  - ko: 'MXSens는 MXINT라는 특정 하드웨어 마이크로스케일링 포맷을 겨냥한다 — 하드웨어 지원이 계속 진화하면서 등장할 다른 블록-스케일 포맷에도 같은 민감도-유도 배분 원칙이 얼마나 쉽게 이식될지는 논문이 다루지 않는다.'
    en: "MXSens targets one specific hardware microscaling format, MXINT — how readily the same sensitivity-guided allocation principle would port to other block-scaled formats that may emerge as hardware support keeps evolving is not addressed in the paper."
source: "autosweep"
---

## Notes

<!-- structured 13-item analysis lives in the frontmatter -->
