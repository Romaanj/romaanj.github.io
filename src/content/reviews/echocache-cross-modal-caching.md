---
title: "EchoCache: Energy-Guided Cross-Modal Caching for Efficient Audio-Driven Video Generation"
arxivId: "2608.02474"
authors: "Jiayu Chen, Xiaoyu Wu, Rongshan Gao, Maoliang Li, Zihao Zheng, Xinhao Sun, Hailong Zou, Guojie Luo, Xiang Chen"
lab: "Peking University"
date: 2026-08-20
tags: ["caching", "video-diffusion", "quantization"]
topic: 'kv-cache'
summary: "Diffusion caching decides what to reuse from visual change alone, which is blind in audio-driven video where the audio says where motion belongs; EchoCache reads audio energy to drive latent-level reuse, reaching 2.46× on Wan2.2-S2V while an int8 residual cache keeps peak memory at 59.12 GB instead of 81.02 GB."
summary_ko: "디퓨전 캐싱은 시각 특징의 변화만 보고 재사용을 결정하는데, 오디오가 어디에 움직임이 있어야 하는지를 이미 말해주는 오디오 구동 영상에서는 그게 눈을 감은 것과 같다. EchoCache는 오디오 에너지로 잠재 단위 재사용을 몰아 Wan2.2-S2V에서 2.46×를 얻고, int8 잔차 캐시로 최대 메모리를 81.02GB 대신 59.12GB에 묶는다."
links: ["coda-nmp-video-diffusion", "liveanimate-streaming-diffusion", "lacache"]
resources:
  - label: "arXiv (abs)"
    url: "https://arxiv.org/abs/2608.02474"
  - label: "arXiv (PDF)"
    url: "https://arxiv.org/pdf/2608.02474"
figures:
  - src: /figures/echocache-cross-modal-caching/fig1.png
    caption: "The four pieces of EchoCache: saliency anchors derived from audio energy, latent-level caching, energy-guided step-latent scheduling, and cache management with adaptive quantization."
    caption_ko: "EchoCache의 네 구성 요소 — 오디오 에너지에서 유도한 saliency anchor, 잠재 단위 캐싱, 에너지 유도 step-latent 스케줄링, 그리고 적응 양자화를 쓰는 캐시 관리."
    credit: "Figure 4 from arXiv:2608.02474 — authors' figure"
  - src: /figures/echocache-cross-modal-caching/fig2.png
    caption: "Where earlier caching methods sit relative to EchoCache for audio-driven video generation: prior work decides reuse per timestep from visual features, EchoCache decides it per latent using the audio."
    caption_ko: "오디오 구동 영상 생성에서 기존 캐싱 기법과 EchoCache의 위치 비교. 앞선 방법들은 시각 특징으로 타임스텝 단위 재사용을 정하고, EchoCache는 오디오로 잠재 단위 재사용을 정한다."
    credit: "Figure 2 from arXiv:2608.02474 — authors' figure"
analysis:
  ko:
    background: '오디오 구동 영상 생성(A2V)은 음성 신호를 조건으로 말하는 얼굴이나 상반신 영상을 만든다. 디퓨전 기반이라 같은 모델을 수십 스텝 반복 호출해야 하고, 그래서 인접 스텝 사이의 중복 계산을 재사용하는 **캐싱**이 이 분야의 표준 가속 수단이 됐다.'
    problem: 'TeaCache·MagCache·TaylorSeer 같은 기존 캐싱은 재사용 여부를 **타임스텝 단위로, 인접 스텝의 시각 특징 변화량만 보고** 결정한다. 그런데 A2V는 본질적으로 교차 모달 문제다 — 어디에 움직임이 있어야 하는지를 오디오가 이미 말해주고 있는데 캐시 결정은 그 신호를 보지 않는다. 저자들은 이로부터 두 가지 어긋남을 짚는다: **시간-의미 어긋남**(오디오가 활발한 구간과 조용한 구간이 똑같은 거친 캐시 처리를 받는다)과 **계산-저장 어긋남**(캐시 할당이 실제로 갱신이 필요한 곳과 맞지 않는다).'
    prior_limits: '이전 방법들의 한계는 정확도가 낮다는 게 아니라 **결정의 해상도와 입력**에 있다. 타임스텝 단위 결정은 한 프레임 안에서 입 주변은 격렬하게 바뀌고 배경은 정지해 있는 상황을 구분하지 못한다. 시각 특징만 보는 것은 이미 손에 쥔 조건 신호를 버리는 셈이다. 실측으로도 이들은 **1.29–1.92×**에 머물면서 품질 지표가 함께 떨어진다.'
    goal: '오디오라는 조건 신호를 캐시 결정에 직접 끌어들여, 재사용 결정의 단위를 타임스텝에서 **잠재 토큰**으로 내리고, 그렇게 잘게 쪼갠 캐시가 메모리를 폭발시키지 않게 만드는 것.'
    method: '입력 오디오를 T개 구간으로 나눠 STFT로 시간-주파수 표현을 얻고, **구간별 시간-주파수 행렬의 L2 노름**으로 에너지 벡터를 만든다. 에너지가 높은 구간이 계산을 더 받아야 할 곳이다. 이 벡터를 잠재 단위 saliency 마스크로 사상한 뒤, 어텐션 계산 전에 **활성 잠재만 뽑아 query 분기를 그들에 대해서만** 실행한다. key와 value는 여전히 전체 잠재 집합에서 만들고, 결과는 `x = m ⊙ x_new + (1−m) ⊙ x_cached`로 융합한다.'
    key_idea: '핵심은 **조건 신호가 곧 계산 예산의 배분표**라는 것이다. 오디오 에너지는 어디서 입과 얼굴이 움직여야 하는지에 대한 사전 정보를 공짜로 들고 있고, 그걸 읽는 비용은 STFT 한 번이다. 여기에 두 번째 아이디어가 붙는다 — 잠재 단위 캐싱은 그대로 두면 캐시 메모리가 구간 수에 비례해 늘어난다. 그래서 유사도 클러스터마다 **전정밀도 기준 캐시** 하나만 두고, 기준과 충분히 가까운(`sim > τ`) 잠재는 **기준 대비 잔차만 int8로** 저장한 뒤 읽을 때 복원한다. 악보 전체를 매번 적는 대신 기준 마디 하나와 그로부터의 차이만 적어두는 셈이다.'
    validation: 'Wan2.2-S2V와 LongCat-Avatar 두 모델을, HDTF와 EMTD 두 데이터셋에서, TeaCache·MagCache·TaylorSeer와 비교한다. 하드웨어는 **NVIDIA H200 141GB**. 지표는 속도뿐 아니라 화질(FID/FVD), 정체성 유지(CSIM), 입-소리 동기(Sync-C), 그리고 **최대 메모리**까지 함께 본다 — 마지막 항목이 이 논문에서 특히 중요하다.'
    results: 'Wan2.2-S2V에서 **2.46×** 가속에 HDTF 기준 FID **66.88** / FVD **117.67**, EMTD 기준 FID **76.01** / FVD **162.66**. LongCat-Avatar에서는 **1.80×**로 HDTF FID **57.82** / FVD **215.75**, EMTD FID **74.85** / FVD **467.12**. 기존 캐싱 기법들은 **1.29–1.92×**에 그치면서 품질 지표가 더 나빴다.'
    comparison: '가장 말이 되는 비교는 절제 실험 쪽에 있다. 오디오 에너지 anchor를 빼고 균일 선택으로 바꾸면 FVD가 **162.66 → 176.05**로 나빠진다 — 즉 이득이 잠재 단위 캐싱 자체가 아니라 **오디오가 고른 위치**에서 나온다는 증거다. 메모리 쪽은 더 결정적이다: 캐시를 FP16 그대로 두면 최대 메모리가 **81.02GB**인데 적응 양자화를 쓰면 **59.12GB**다. 반대로 순진하게 int8로 밀면 FVD가 **183.26**까지 나빠진다. 잔차 기준 양자화라는 형태가 필요했다는 뜻이다.'
    significance: '효율 관점에서 이 논문이 옮겨놓은 것은 **캐시 결정의 입력**이다. 캐싱 문헌은 대체로 자기 자신의 활성값 변화를 보고 재사용을 판단해왔는데, 조건부 생성에서는 조건 자체가 더 싸고 더 이른 신호다. 그리고 메모리 절제 결과가 보여주듯, 캐시 입도를 잘게 만드는 개입은 **저장 형식을 함께 설계하지 않으면 지연을 사고 메모리를 잃는 거래**가 된다 — FP16 변형은 캐시를 안 쓴 기준선(57.23GB)보다 메모리를 더 쓴다.'
    limitations: '저자들의 서술 그대로: 오디오 에너지 사전정보가 더 풍부한 교차 모달 단서를 온전히 담지 못할 수 있고, 캐시 스케줄링이 모델과 설정에 따라 **여전히 부분적으로 휴리스틱**이다. 덧붙이면 에너지는 진폭 대리지표라 조용한 자음이나 무성 구간처럼 소리는 작지만 입 모양 변화가 큰 경우를 놓칠 수 있다(리뷰어 판단).'
    future_work: '저자들은 영상 입력이나 궤적 같은 다른 움직임 관련 조건 양식으로 확장해 더 일반적인 양식 인지 캐싱 프레임워크로 가는 방향을 제시한다.'
    resources: '논문은 `github.com/IF-LAB-PKU/EchoCache`에 코드가 있다고 밝히지만 확인 시점에 해당 저장소가 열리지 않았다(404) — 공개 링크 확인 안 됨. 검증된 1차 출처는 arXiv 초록과 PDF뿐이다.'
  en:
    background: 'Audio-driven video generation (A2V) synthesizes a talking face or upper body conditioned on a speech signal. Because it is diffusion-based, the same model is invoked over dozens of steps, which is why **caching** — reusing computation that barely changes between adjacent steps — became the standard accelerator in this area.'
    problem: 'Existing caching methods such as TeaCache, MagCache and TaylorSeer decide reuse **at the timestep level, from the change in visual features alone**. But A2V is inherently cross-modal: the audio already says where motion belongs, and the cache decision never looks at it. The authors name two misalignments that follow — **temporal-semantic** (audio-active and audio-quiet stretches receive the same coarse cache treatment) and **computation-storage** (cache allocation does not match where updates are actually needed).'
    prior_limits: 'What limits the prior methods is not their accuracy but the **resolution and the input of the decision**. A timestep-level choice cannot separate a frame in which the mouth region churns while the background sits still. Looking only at visual features discards a conditioning signal already in hand. Measured, these methods stop at **1.29–1.92×** and give up quality while doing so.'
    goal: 'Pull the audio conditioning signal directly into the cache decision, lower the granularity of reuse from the timestep to the **latent token**, and keep the resulting fine-grained cache from exploding memory.'
    method: 'The input audio is split into T segments and converted to a time-frequency representation by STFT; the **L2 norm of each segment''s time-frequency matrix** gives an energy vector, with high-energy segments marking where computation should go. That vector is mapped to a latent-level saliency mask, and before attention the model **extracts the active latents and runs the query branch only on those**. Keys and values are still computed from the complete latent set, and the results fuse as `x = m ⊙ x_new + (1−m) ⊙ x_cached`.'
    key_idea: 'The central move is treating **the conditioning signal as the compute budget allocator**. Audio energy carries free prior information about where the mouth and face must move, and reading it costs one STFT. A second idea follows from the first: latent-level caching would otherwise grow cache memory in proportion to the number of cached segments. So each similarity cluster keeps one **full-precision reference cache**, and any latent close enough to it (`sim > τ`) stores only **its int8-quantized residual against that reference**, reconstructed on read. It is the difference between rewriting a full score every time and noting one reference bar plus the deviations from it.'
    validation: 'Two models, Wan2.2-S2V and LongCat-Avatar, are evaluated on two datasets, HDTF and EMTD, against TeaCache, MagCache and TaylorSeer, on an **NVIDIA H200 141GB**. The metrics cover more than speed — image quality (FID/FVD), identity preservation (CSIM), lip-sync (Sync-C), and **peak memory**, the last of which turns out to matter unusually much here.'
    results: 'On Wan2.2-S2V, **2.46×** speedup at FID **66.88** / FVD **117.67** on HDTF and FID **76.01** / FVD **162.66** on EMTD. On LongCat-Avatar, **1.80×** at FID **57.82** / FVD **215.75** on HDTF and FID **74.85** / FVD **467.12** on EMTD. The prior caching methods reach only **1.29–1.92×** and score worse on quality while doing so.'
    comparison: 'The most informative comparison is internal, in the ablations. Removing the audio-energy anchors in favour of uniform selection degrades FVD from **162.66 to 176.05** — evidence that the gain comes from **where the audio points**, not from latent-level caching by itself. The memory ablation is sharper still: leaving the cache in FP16 costs **81.02 GB** of peak memory against **59.12 GB** with adaptive quantization, while quantizing naively to int8 degrades FVD to **183.26**. The residual-against-reference form was doing necessary work.'
    significance: 'Through an efficiency lens, what this paper relocates is **the input to the cache decision**. The caching literature has largely judged reuse by watching its own activations change, but in conditional generation the condition itself is the cheaper and earlier signal. And as the memory ablation shows, any intervention that makes cache granularity finer is **a trade that buys latency and loses memory unless the storage format is designed alongside it** — the FP16 variant consumes more memory than the uncached baseline at 57.23 GB.'
    limitations: 'In the authors'' own words, the audio-energy prior may not fully capture richer cross-modal cues, and the cache scheduling remains **partly heuristic across models and settings**. One more exposed edge: energy is a proxy for amplitude, so quiet consonants and unvoiced stretches — low in sound, large in mouth-shape change — are the cases most likely to be missed (reviewer judgment).'
    future_work: 'The authors point toward broader multimodal conditions, such as video inputs and other motion-related trajectory modalities, as the route to a more general modality-aware caching framework.'
    resources: 'The paper states that code is available at `github.com/IF-LAB-PKU/EchoCache`, but that repository did not resolve at the time of checking (404) — no public release verified. The arXiv abstract and PDF are the only verified primary sources.'
thread:
  ko: |-
    디퓨전 가속에서 캐싱 계열은 하나의 관찰 위에 서 있다 — 인접한 디노이징 스텝의 활성값은 서로 매우 닮아 있으므로, 그 차이가 작을 때는 다시 계산하지 말고 재사용하라는 것. TeaCache부터 MagCache, TaylorSeer까지 이 줄기의 발전은 대체로 "얼마나 닮았는지를 더 잘 재는 법"에 관한 것이었다. 지표는 정교해졌지만 재사용 결정의 단위는 타임스텝에 머물렀고, 입력은 언제나 모델 자신의 활성값이었다.

    EchoCache가 바꾸는 지점은 지표가 아니라 **결정에 쓰는 신호의 출처**다. 조건부 생성에서는 조건 자체가 무엇이 변할지에 대한 정보를 이미 들고 있다 — 오디오가 조용하면 입은 움직이지 않는다. 그러면 활성값이 변하는 것을 관찰해서 뒤늦게 알아내는 대신, 조건에서 미리 읽어낼 수 있다. 이 전환이 자연스럽게 결정 단위를 잠재 토큰까지 내린다. 오디오는 프레임 안 어디가 움직일지까지 말해주기 때문이다.

    그런데 이 논문에서 가장 오래 남을 교훈은 아마 메모리 절제 실험일 것이다. 캐시를 잘게 쪼개면 지연은 줄지만 저장할 조각이 늘어난다 — FP16으로 두면 캐시를 아예 안 쓴 것보다 메모리를 더 쓴다. 입도를 높이는 모든 캐싱 작업이 같은 벽을 만나게 되고, 그래서 "무엇을 재사용할지"와 "재사용할 것을 어떤 형식으로 들고 있을지"는 사실 하나의 설계 문제다. 조건 신호로 캐시를 모는 아이디어가 다른 양식(영상, 궤적, 텍스트 조건)으로 번져갈수록, 이 저장 형식 쪽 질문이 함께 커질 것이다.
  en: |-
    The caching line in diffusion acceleration rests on one observation: activations at adjacent denoising steps look very much alike, so when the difference is small, reuse instead of recompute. From TeaCache through MagCache and TaylorSeer, progress along this line has mostly been about measuring that likeness better. The metrics grew more refined, but the unit of the reuse decision stayed the timestep, and the input was always the model's own activations.

    What EchoCache changes is not the metric but **where the signal for the decision comes from**. In conditional generation the condition already carries information about what is going to change — if the audio is quiet, the mouth is not moving. Rather than discovering that after the fact by watching activations shift, it can be read from the condition in advance. That shift naturally lowers the decision to the latent token, because the audio says not just when but where within the frame motion belongs.

    The lesson likeliest to outlive the paper, though, may be the memory ablation. Making a cache finer-grained reduces latency but multiplies the pieces to store — left in FP16, it costs more memory than not caching at all. Every caching method that raises granularity walks into the same wall, which means "what to reuse" and "in what format to hold what is reused" are really one design problem. As the idea of steering caches by the conditioning signal spreads to other modalities — video, trajectories, text conditions — this storage-side question will grow along with it.
sparks:
  - ko: "저자들이 오디오 에너지가 풍부한 교차 모달 단서를 다 담지 못한다고 인정한다. 진폭 대신 음소 단위나 조음 특징으로 anchor를 만들면, 소리는 작지만 입 모양이 크게 바뀌는 구간에서 얼마나 달라질까?"
    en: "The authors concede that audio energy does not capture richer cross-modal cues. If the anchor came from phoneme or articulatory features instead of amplitude, how much would change in exactly the stretches that are quiet in sound but large in mouth-shape?"
  - ko: "스케줄링이 모델·설정마다 부분적으로 휴리스틱이라고 밝힌다. 임계값 τ와 top-K를 손으로 정하지 않고 캐시 오차 예산으로부터 유도할 수 있다면, 모델이 바뀔 때마다 다시 맞추는 일이 사라질까?"
    en: "Scheduling is described as partly heuristic across models and settings. If the threshold τ and top-K were derived from a cache-error budget rather than hand-set, would the per-model retuning disappear?"
  - ko: "FP16 캐시가 무캐시 기준선보다 메모리를 더 쓴다는 결과는 입도를 높이는 모든 캐싱에 해당될 법하다. 다른 디퓨전 캐싱 기법들도 같은 입도로 밀면 같은 벽을 만나는지, 그리고 잔차 양자화가 거기서도 통하는지는 열려 있다."
    en: "That an FP16 cache costs more memory than not caching at all looks like it should generalize to any method that raises granularity. Whether other diffusion caching methods hit the same wall when pushed to the same granularity — and whether residual quantization rescues them too — is open."
source: "autosweep"
---

## Notes

<!-- The structured 13-item analysis lives in the frontmatter. -->
