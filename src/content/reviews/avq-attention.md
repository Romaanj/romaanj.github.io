---
title: "AVQ-Attention: Adaptive Vector-Quantized Attention"
arxivId: "2607.12789"
authors: "Winfried van den Dool, Patrick Forré, Amir Habibian, Yuki M. Asano, Max Welling"
date: 2026-07-15
tags: ["attention", "vector-quantization", "efficiency", "vision"]
topic: 'architecture'
summary: "Vector-quantized attention cuts quadratic attention cost by clustering keys into a fixed codebook, but spends the same resolution everywhere regardless of where attention mass actually concentrates; AVQ-Attention adaptively refines codebook resolution toward high-attention regions using a hierarchical parent-child codebook, implemented as custom Triton kernels fused directly into Flash-Attention-style tiling, reaching up to 127x over Flash Attention at 64k tokens."
summary_ko: "벡터-양자화 attention은 key를 고정된 코드북으로 클러스터링해 이차 비용을 줄이지만 attention이 실제로 집중되는 곳과 무관하게 어디서나 같은 해상도를 쓴다; AVQ-Attention은 계층적 parent-child 코드북으로 고-attention 영역에만 해상도를 적응적으로 정밀화하고, 이를 Flash-Attention 스타일 타일링에 직접 융합한 커스텀 Triton 커널로 구현해 64k 토큰에서 Flash Attention 대비 최대 127배에 이른다."
links: ['duo-attention', 'going-linear', 'turboquant']
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2607.12789' }
figures:
  - src: /figures/avq-attention/fig1.png
    caption: "Key-space visualization of adaptive VQ-attention: keys assigned to codewords, per-codeword importance computed from received attention mass, then the top parents' children spawned for finer resolution where it matters."
    caption_ko: "적응형 VQ-attention의 키 공간 시각화: 키를 코드워드에 할당하고, 받은 attention 질량으로 코드워드별 중요도를 계산한 뒤, 상위 parent의 child를 spawn해 필요한 곳에만 해상도를 높인다."
    credit: "Figure 1 from arXiv:2607.12789 — authors' figure"
analysis:
  ko:
    background: 'Attention의 O(N²) 연산 비용은 시퀀스 길이 N이 커질수록 transformer의 근본적인 병목이 되며, Flash Attention 같은 타일링 기법은 메모리 대역폭 문제는 해결해도 연산량 자체는 그대로 이차로 남긴다. 벡터-양자화(VQ) attention은 key를 M개의 대표 코드워드로 클러스터링해 비용을 O(MN)으로 낮추는 대안을 제시해 왔다.'
    problem: '기존 VQ-attention은 attention 질량이 실제로 어디에 몰려 있는지와 무관하게 모든 key 공간 영역에 동일한 코드북 용량을 배정한다. 그 결과 attention이 집중되는 영역은 거칠게 근사되어 정확도를 잃고, attention이 거의 없는 영역은 불필요하게 세밀한 표현 용량을 낭비한다.'
    prior_limits: '고정-코드북 VQ-attention은 코드북 크기 M을 늘려 정확도를 높일 수 있지만, 그러면 관심 없는 영역까지 균일하게 세밀해져 O(MN) 비용이 목표 없이 커진다. 희소 attention 패턴이나 토큰 병합 같은 다른 효율화 기법들도 대체로 구조를 정적으로 고정해, 어디에 중요한 attention이 몰릴지는 반영하지 않는다.'
    goal: '어떤 재학습도 없이 추론 시점에 속도-정확도를 조절할 수 있으면서, attention이 실제로 집중되는 key 공간 영역에만 세밀한 양자화를 배정하고 나머지는 거친 양자화로 남겨 두는 VQ-attention을 설계하는 것이다.'
    method: '작은 parent 코드워드 집합에서 시작해, 순전파 도중 각 코드워드가 받는 attention 질량으로 중요도를 계산하고, 가장 중요한 상위 𝒫개 parent에 대해서만 미리 학습된 𝒞개의 child 코드워드를 "spawn"해 그 영역의 해상도를 정밀화한다. 이 전 과정 — 중요도 계산, child spawn, parent-기여 대체 — 을 Flash Attention 스타일의 타일 계산 루프 안에 직접 넣는 커스텀 Triton 커널로 구현해, 추가 자료 구조를 SRAM 밖으로 유출시키지 않는다.'
    key_idea: '핵심은 코드북 해상도 자체를 추론 시점에, 별도 재학습 없이 attention 중요도에 따라 적응시킨다는 것이다 — 마치 지도를 그릴 때 사람이 몰리는 도심은 골목까지 세밀하게, 인적 드문 외곽은 큰 구획으로만 그리는 것과 비슷하다. 𝒫를 바꾸는 것만으로 재학습 없이 하나의 체크포인트에서 속도와 정확도를 오가는 다이얼을 얻는다는 점이 실질적 이점이다.'
    validation: 'ImageNet-1k 분류(ViT-Base)와 ADE20K 시맨틱 분할(DPT-Large)에서 사전학습된 transformer의 attention 층을 교체해 짧게 미세조정한 뒤 평가하며, Stable Diffusion 1.5 UNet의 self-attention에 LinFusion 증류 프로토콜을 그대로 적용해 검증한다. N=65,536까지 실제 wall-clock 커널 시간을 측정해 이론적 선형 스케일링을 확인하고, VQ 사전계산 단계를 융합하는 것만으로 얻는 이득과 적응형 해상도 배분이 추가로 주는 이득을 분리해서 보고한다.'
    results: 'N=65,536에서 융합된 AVQ-attention 설정은 Flash Attention 대비 81-127배 빠르며, flat VQ의 경우도 VQ 사전계산을 커널에 융합하는 것만으로 약 2배의 속도 향상을 얻는다. ImageNet-1k·ADE20K 모두에서 동일 attention 커널 비용 대비 flat VQ-attention보다 높은 정확도를 얻고(ADE20K mIoU 43.33% vs flat-VQ 42.70%, Swin 42.90%, NATTEN 40.99%), Stable Diffusion 1.5에서는 1024² 미학습 해상도로 확장해도 LinFusion·ToMe-SD보다 낮은 UNet forward 시간에 대등하거나 나은 FID·CLIP 점수를 유지한다.'
    comparison: '동일 커널 비용에서 flat VQ-attention·Swin·NATTEN·Linformer·Performer를 모두 앞서며, 특히 Performer는 Flash-Attention과 호환되지 않아 실질적 비교에서 더 불리하다. Flash Attention 자체와 비교하면 VQ라는 아이디어와 그것을 타일 루프에 완전히 융합하는 엔지니어링이 기여를 나눠 갖는데, 저자들은 이를 분리해서 "융합 자체의 이득"과 "적응형 배분의 이득"을 각각 보고한다는 점에서 비교가 정직하다.'
    significance: '효율적 AI 관점에서 이 논문은 "중요도에 따라 표현 용량을 적응적으로 배분한다"는 원리가 커널 융합과 결합될 때 추가 오버헤드 없이 공짜로 얻어질 수 있음을 보여준다 — 이는 attention 근사 전반에 적용 가능한 일반적 설계 원리로서, KV 캐시 압축을 포함한 다른 적응형 할당 방법들의 참고 사례가 된다. 다만 이 논문 자체는 인과적 언어 모델링이나 KV 캐시 세팅을 전혀 다루지 않는다는 점은 분명히 해 둘 필요가 있다.'
    limitations: '𝑀₀·𝒫·𝒞의 최적값이 층마다 다를 가능성이 높다고 저자들 스스로 언급하지만("일부 층은 훨씬 적은 코드워드만 필요할 수 있다"), 층별 자동 설정은 시도되지 않고 향후 과제로 남아 있다. 코드북은 사전학습된 transformer를 몇 epoch 미세조정해 얻어야 하므로 완전한 training-free 방법은 아니며, 2단계 이상의 더 깊은 계층 구조는 제안만 되고 실험되지 않았다.'
    future_work: '저자들은 캡처된 attention 질량과 양자화 오차라는, AVQ-attention이 이미 노출하는 두 신호를 이용해 재학습 없이 층별 아키텍처 탐색을 할 수 있으리라 제안한다. 또한 계층을 한 단계 더 깊게 하면 추가 비용은 선형(𝒫𝒞)인 반면 코드북 해상도는 곱셈적으로 늘어난다는 점을 지적하며, 이를 다음 자연스러운 확장으로 남겨 둔다.'
    resources: '논문 자체에 별도의 코드 저장소 링크는 확인되지 않았다 — arXiv 게재 외 공개 구현체는 발견되지 않았다.'
  en:
    background: 'The O(N²) cost of attention over N tokens remains a fundamental bottleneck as sequences scale, and while Flash Attention''s tiling solves the memory-bandwidth side of the problem, the underlying computation stays quadratic. Vector-Quantized (VQ) attention offers an alternative, clustering keys into M representative codewords to cut cost to O(MN).'
    problem: 'Existing VQ-attention applies the same codebook capacity everywhere in key space regardless of where attention mass actually concentrates. High-attention regions end up coarsely approximated and lose accuracy, while low-attention regions receive representational capacity they do not need.'
    prior_limits: 'A fixed-codebook VQ-attention can only trade accuracy for cost by growing M uniformly, which wastes resolution on regions nobody attends to and grows O(MN) cost without regard to where it actually helps. Other efficient-attention approaches -- sparse patterns, token merging -- likewise tend to fix their structure statically rather than adapting to where attention mass concentrates.'
    goal: 'Design a VQ-attention variant that concentrates fine-grained quantization exactly where attention mass concentrates and leaves the rest coarse, while allowing the speed/accuracy tradeoff to be adjusted at inference time with no retraining.'
    method: 'Starting from a small set of parent codewords, the method computes per-codeword importance from the attention mass each parent receives during the actual forward pass, then "spawns" pre-learned child codewords for only the top-𝒫 most important parents -- refining resolution exactly where it matters and leaving everything else at the coarser parent level. The entire process (importance scoring, child spawning, parent-to-child contribution replacement) is implemented in custom Triton kernels fused directly into a Flash-Attention-style tiled computation loop, so nothing needs to leave SRAM.'
    key_idea: 'The central move is adapting codebook resolution itself, at inference time and with no retraining, to where attention importance actually lies -- much like a cartographer drawing a city map in fine street-level detail downtown while rendering sparsely visited outskirts as large blocks. Changing 𝒫 alone, with no retraining, turns one trained checkpoint into a speed/accuracy dial, which is the practical payoff.'
    validation: 'Evaluated by replacing attention layers in pretrained transformers and fine-tuning briefly, on ImageNet-1k classification (ViT-Base) and ADE20K semantic segmentation (DPT-Large), and by applying the same substitution to Stable Diffusion 1.5''s UNet self-attention under the LinFusion distillation protocol. Wall-clock kernel time is measured out to N=65,536 to confirm the predicted linear scaling, with the gain from fusing the VQ precompute step reported separately from the additional gain from adaptive resolution allocation.'
    results: 'At N=65,536, fused AVQ-attention configurations are 81-127x faster than Flash Attention, and fusing the VQ precompute step alone (even for flat, non-adaptive VQ) already buys roughly 2x. AVQ beats flat VQ-attention at matched kernel cost on both ImageNet-1k and ADE20K (ADE20K mIoU 43.33% vs. flat-VQ 42.70%, Swin 42.90%, NATTEN 40.99%), and on Stable Diffusion 1.5, extended to an unseen 1024² resolution, it matches or beats LinFusion and ToMe-SD on FID/CLIP at a lower UNet forward time.'
    comparison: 'At matched kernel cost, it outperforms flat VQ-attention, Swin, NATTEN, Linformer, and Performer (the last of which is additionally not Flash-Attention-compatible, a further practical disadvantage). Against Flash Attention itself, the reported speedup is shared between the VQ idea and the engineering effort of fusing it fully into the tile loop -- and the paper is honest about this, reporting the fusion-only gain and the adaptive-allocation gain separately rather than conflating them into one headline number.'
    significance: 'From an efficient-AI standpoint, this shows that "allocate representational capacity by importance" can be delivered with no extra kernel-level overhead once flat VQ is already fused into a tiled loop -- a general design principle applicable to attention approximation broadly, and a useful reference point for other adaptive-allocation methods, including KV-cache compression. It is worth being explicit, though, that the paper itself says nothing about causal language modeling or a cached, cross-step KV-cache setting.'
    limitations: 'The authors themselves note the optimal M₀/𝒫/𝒞 likely varies by layer ("some layers may need far fewer codewords"), but automatic per-layer configuration is not attempted and is left as future work. Codebooks require a short fine-tuning pass on a pretrained transformer rather than being usable training-free, and deeper (more than one level) hierarchies are proposed but never tested.'
    future_work: 'The authors suggest that captured attention mass and quantization error -- two signals AVQ-attention already exposes -- could enable layer-wise architecture search without full retraining. They also note that adding one more level of hierarchy costs only a linear (𝒫𝒞) increment while multiplying codebook resolution, and leave this as the natural next extension.'
    resources: 'No separate code repository link was found in the paper -- no public implementation beyond the arXiv listing was located.'
thread:
  ko: |-
    효율적 attention 근사는 오랫동안 여러 갈래로 갈라져 발전해 왔다 — Longformer·희소 패턴류가 어떤 토큰끼리 상호작용할지 정적으로 제한하는 쪽이라면, ToMe 같은 토큰 병합은 토큰 수 자체를 줄이는 쪽이고, VQ-attention은 key를 대표 코드워드로 클러스터링해 O(MN)으로 낮추는 세 번째 갈래였다. 이 VQ 계열은 가능성을 보여주면서도 늘 하나의 미해결 문제를 남겼다: 코드북을 어디에 세밀하게 쓸지는 항상 고정되어 있었다는 점이다.

    이 논문의 개념적 전환은 코드북 해상도 자체를 순전파 도중 관측된 attention 질량에 따라 적응시킨다는 것이다 — parent-child 계층 구조로 중요한 영역만 정밀화하고, 그 전체 과정을 Flash Attention 타일 루프 안에 완전히 융합해 넣는다. 이는 단순히 "VQ에 적응성을 추가한다"는 아이디어 차원의 기여를 넘어, 그 적응성이 이미 존재하는 커널 융합 파이프라인 위에 사실상 무료로 얹힐 수 있음을 엔지니어링으로 증명한 것에 가깝다.

    이 논문이 스스로 열어두는 다음 단계는 층별 자동 설정(캡처된 attention 질량과 양자화 오차라는 이미 노출된 신호를 이용한 재학습 없는 아키텍처 탐색)과 더 깊은 계층 구조다. 더 흥미로운 열린 질문은 논문 밖에 있다: 이 논문의 모든 실험은 비인과적(non-causal), 매 순전파마다 다시 계산되는 attention(분류·분할·확산 UNet)에 한정되어 있어, 인과적 언어모델 디코딩이나 여러 스텝에 걸쳐 재사용되는 KV 캐시 세팅에 "중요도에 따라 적응적으로 배분한다"는 동일한 원리가 어떻게 이식될 수 있을지는 전혀 다루어지지 않은 채 남아 있다.
  en: |-
    Efficient attention approximation has long split into several separate lines -- sparse-pattern methods like Longformer statically restrict which tokens interact, token-merging methods like ToMe reduce the number of tokens processed, and VQ-attention forms a third line, clustering keys into representative codewords to reach O(MN) cost. This VQ line showed promise but always left one thing unresolved: where the codebook spent its fine resolution was fixed in advance.

    This paper's conceptual shift is to adapt codebook resolution itself to the attention mass actually observed during the forward pass -- refining only the important regions via a parent-child hierarchy, and fusing that entire process directly into the Flash Attention tile loop. This is less a purely algorithmic contribution ("add adaptivity to VQ") than an engineering demonstration that this adaptivity can ride on top of an already-fused kernel pipeline at essentially no extra cost.

    The paper names its own next steps: automatic per-layer configuration (using the attention-mass and quantization-error signals it already exposes, without full retraining) and deeper hierarchies. The more interesting open question sits outside the paper itself: every experiment here is on non-causal attention recomputed at every forward pass (classification, segmentation, diffusion UNets), leaving entirely untouched how the same "allocate adaptively by importance" principle would need to change to reach causal language-model decoding and a KV cache reused across many steps.
sparks:
  - ko: "저자들이 명시적으로 제안한 대로, 이 방법이 이미 노출하는 두 신호(캡처된 attention 질량, 양자화 오차)를 이용해 재학습 없이 층별로 𝑀₀·𝒫·𝒞를 자동 선택하는 절차를 설계해볼 수 있다."
    en: "As the authors explicitly suggest, one could design a procedure that automatically selects per-layer M₀/𝒫/𝒞 using the two signals the method already exposes (captured attention mass, quantization error), without any full retraining."
  - ko: "저자들이 제안만 하고 실험하지 않은 2단계 이상의 더 깊은 parent-child 계층을 실제로 구현해, 선형 비용 증가(𝒫𝒞)로 코드북 해상도가 정말 곱셈적으로 늘어나는지, 그리고 어느 깊이에서 이득이 정체되는지 측정해볼 수 있다."
    en: "One could actually implement the deeper (more than one level) parent-child hierarchy the authors propose but never test, measuring whether codebook resolution really multiplies for a linear (𝒫𝒞) cost increment, and at what depth the gains plateau."
source: "autosweep"
---

## Notes

The paper is careful to separate two contributions that a less rigorous write-up might conflate into one headline speedup number: fusing the VQ precompute step into the kernel (which alone buys ~2x, even for a flat, non-adaptive codebook) and the additional accuracy gained from making resolution adaptive (measured at matched kernel cost against flat VQ). That separation is what makes the 81-127x Flash-Attention comparison legible rather than a single conflated number.
