---
part: 2
title: "TPU를 이해하는 법"
title_en: "How to Think About TPUs"
original: "https://jax-ml.github.io/scaling-book/tpus/"
summary: "TPU의 내부 구조(TensorCore, MXU, VPU, VMEM, HBM)와 TPU들이 ICI·DCN으로 연결되어 pod를 이루는 방식, 그리고 이 구조가 멀티칩 학습·추론 성능에 미치는 영향을 다룬다. 부록에서는 VPU·scalar core의 내부와 systolic array의 동작 원리까지 파고든다."
date: 2026-08-20
published: true
---

> 이 장은 TPU가 어떻게 동작하는지, 멀티칩 학습과 추론이 가능하도록 TPU들이 어떻게 네트워크로 연결되는지, 그리고 이것이 우리가 아끼는 알고리즘들의 성능에 어떤 영향을 주는지를 다룬다. GPU 사용자에게도 좋은 내용이 있다!

<div class="takeaway">

NVIDIA GPU를 다루는 새 [12장](/scaling-book/gpus/)도 재미있게 읽을 수 있을 것이다!

</div>

## TPU란 무엇인가?

**TPU는 기본적으로 행렬 곱셈에 특화된 연산 코어(TensorCore)에 빠른 메모리 스택(high-bandwidth memory, HBM)이 붙어 있는 구조다** (Jouppi et al. 2023). 다이어그램으로 보면 다음과 같다:

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/tpu-chip.png" alt="TPU 칩의 기본 구성 요소" loading="lazy" />
  <figcaption><b>그림:</b> TPU 칩의 기본 구성 요소. TensorCore는 왼쪽 회색 박스로, matrix-multiply unit(MXU), vector unit(VPU), vector memory(VMEM)를 담고 있다.</figcaption>
</figure>

TensorCore는 기본적으로 아주 뛰어난 행렬 곱셈 기계라고 생각하면 되지만, 그 밖에도 주목할 만한 기능이 몇 가지 있다. TensorCore에는 세 가지 핵심 유닛이 있다:

* **MXU**(Matrix Multiply Unit)는 TensorCore의 핵심이다. 대부분의 TPU 세대에서 systolic array를 이용해 8 사이클마다 한 번의 `bf16[8,128] @ bf16[128,128] -> f32[8,128]` 행렬 곱셈을 수행한다[^1] (자세한 내용은 아래 부록 B 참조).
  * TPU v5e에서 1.5GHz 기준으로 MXU당 약 `5e13` bf16 FLOPs/s다. 대부분의 TensorCore는 MXU를 2개 또는 4개 갖고 있어서, 예컨대 TPU v5e의 총 bf16 FLOPs/s는 `2e14`다.
  * TPU는 더 낮은 정밀도의 matmul을 더 높은 throughput으로 지원하기도 한다(예: TPU v5e 칩 하나는 `4e14` int8 OPs/s를 수행할 수 있다).

* **VPU**(Vector Processing Unit)는 ReLU activation이나 벡터 간 원소별(pointwise) 덧셈·곱셈 같은 일반적인 수학 연산을 수행한다. reduction(합산)도 여기서 수행된다. 자세한 내용은 아래 부록 A에 있다.
* **VMEM**(Vector Memory)은 TensorCore 안, 연산 유닛 가까이에 위치한 온칩 scratchpad다. HBM보다 훨씬 작지만(예: TPU v5e에서 128 MiB) MXU로의 bandwidth는 훨씬 높다. VMEM은 CPU의 L1/L2 캐시와 어느 정도 비슷하게 동작하지만 훨씬 크고 프로그래머가 직접 제어한다. HBM에 있는 데이터는 TensorCore가 연산에 쓰기 전에 VMEM으로 복사되어야 한다.

**TPU는 행렬 곱셈이 매우, 매우 빠르다.** 주로 하는 일이 그것이고, 잘한다. 역대 가장 강력한 TPU 중 하나인 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)는 코어당 `2.5e14` bf16 FLOPs/s, 칩당 `5e14` bf16 FLOPs/s를 수행할 수 있다. 8960개 칩으로 이루어진 pod 하나면 초당 4 bf16 exaFLOPs다. *엄청난* 양이다. 세계에서 가장 강력한 슈퍼컴퓨터 중 하나이고, Google은 이걸 많이 갖고 있다.[^2]

위 다이어그램에는 SMEM이나 scalar unit 같은 다른 구성 요소도 몇 개 있는데, 이들은 제어 흐름 처리에 쓰이며 부록 A에서 간단히 다루지만 꼭 이해해야 하는 것은 아니다. 반면 HBM은 중요하면서도 꽤 단순하다:

* **HBM**(High Bandwidth Memory)은 TensorCore가 사용할 텐서들을 저장하는 큰 고속 메모리다. HBM 용량은 보통 수십 기가바이트 수준이다(예: [TPU v5e는 16GiB의 HBM을 갖는다](https://cloud.google.com/tpu/docs/v5e#system_architecture)).

  * 연산에 필요해지면 텐서는 HBM에서 VMEM을 거쳐(아래 참조) MXU로 스트리밍되고, 결과는 VMEM에서 다시 HBM으로 쓰인다.

  * HBM과 TensorCore 사이(VMEM 경유)의 bandwidth를 "HBM bandwidth"라고 부르며(보통 1-2TB/sec 수준), memory-bound 워크로드에서 연산이 얼마나 빨리 수행될 수 있는지를 제한한다.

**일반적으로 모든 TPU 연산은 pipeline되고 겹쳐진다(overlap).** matmul $X \cdot A \to Y$를 수행하려면 TPU는 먼저 행렬 $A$와 $X$의 청크를 HBM에서 VMEM으로 복사하고, 이를 MXU에 로드해 ($X$의) 8x128 청크와 ($A$의) 128x128 청크를 곱한 다음, 결과를 청크 단위로 다시 HBM에 복사해야 한다. 이를 효율적으로 하기 위해 matmul을 pipeline해서 VMEM을 오가는 복사가 MXU 작업과 겹치게 만든다. 그러면 MXU가 메모리 전송을 기다리는 대신 계속 일할 수 있어, matmul이 memory-bound가 아니라 compute-bound로 유지된다.

HBM에서 원소별 곱(elementwise product)을 수행하는 방식의 예시:

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/pointwise-product.gif" alt="TPU에서의 pointwise product 애니메이션" loading="lazy" />
  <figcaption><b>그림:</b> TPU에서 HBM으로부터 바이트를 로드하며 수행되는 pointwise product를 보여주는 애니메이션. 바이트가 메모리에서 청크 단위로 스트리밍되어 나오고, 전체 배열이 다 만들어지기를 기다리지 않고 부분 결과가 pipeline되어 되돌아가는 점에 주목하라.</figcaption>
</figure>

matmul도 거의 똑같아 보이겠지만, VPU/vector unit 대신 MXU로 로드된다는 점과, 같은 weight 청크가 여러 activation 청크에 쓰이므로 로드와 스토어가 다른 순서로 일어난다는 점이 다르다. 데이터 청크가 VMEM으로, 그다음 VREG(vector register)로, 그다음 Vector Unit으로 흘러 들어갔다가, 다시 VMEM과 HBM으로 돌아오는 것을 볼 수 있다. 곧 보겠지만, HBM에서 VMEM으로의 로드가 Vector Unit(또는 MXU)의 FLOPs보다 느리면 VPU나 MXU에 일감이 부족해져 "bandwidth bound" 상태가 된다.

<div class="takeaway">

**요점(Takeaway):** TPU는 매우 단순하다. weight를 HBM에서 VMEM으로 로드하고, VMEM에서 초당 약 200조 번의 곱셈-덧셈을 수행할 수 있는 systolic array로 로드한다. HBM $\leftrightarrow$ VMEM과 VMEM $\leftrightarrow$ systolic array bandwidth가 TPU가 효율적으로 수행할 수 있는 연산의 근본적인 한계를 결정한다.

</div>

**VMEM과 arithmetic intensity:** VMEM은 HBM보다 훨씬 작지만 MXU로의 bandwidth는 훨씬 높다. [1장](/scaling-book/roofline/)에서 봤듯이, 알고리즘의 입력/출력을 전부 VMEM에 담을 수 있으면 통신 병목에 걸릴 가능성이 훨씬 낮아진다. 연산의 arithmetic intensity가 나쁠 때 특히 도움이 된다: VMEM bandwidth는 HBM bandwidth보다 약 22배 높아서, VMEM에서 읽고 VMEM에 쓰는 MXU 연산은 arithmetic intensity가 10-20만 되어도 peak FLOPs 활용에 도달한다. weight를 HBM 대신 VMEM에 넣을 수 있다면 행렬 곱셈은 훨씬 작은 batch size에서도 FLOPs bound가 된다. 그리고 근본적으로 arithmetic intensity가 낮은 알고리즘도 여전히 효율적일 수 있다. 다만 VMEM이 워낙 작아서 이게 쉽지 않은 경우가 많다.[^3]

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/tpu-bandwidth.png" alt="TPU 메모리 계층별 bandwidth" loading="lazy" />
</figure>

**TPU 칩은 보통(항상은 아니지만) 메모리를 공유하는 두 개의 TPU 코어로 구성되며, FLOPs가 두 배인 하나의 큰 가속기로 생각할 수 있다**("megacore" 구성이라고 부른다). v4, v5, v6 TPU가 그렇다(TPU v7은 megacore를 없애는 대신 두 코어 사이에 high-bandwidth 링크를 둔다). 더 오래된 TPU 칩은 메모리가 분리되어 있어 두 개의 별도 가속기로 취급된다(TPU v3 이전). TPU v5e 같은 추론 최적화 칩은 칩당 TPU 코어가 하나뿐이다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/cores.png" alt="TPU 코어 구성" class="img-small" loading="lazy" />
</figure>

**칩**은 **'tray' 위에 4개 단위로** 배치되어 **PCIe 네트워크로 CPU host에 연결된다.** 대부분의 독자에게 익숙한 형태로, Colab이나 단일 TPU-VM을 통해 노출되는 것이 바로 이 4개 칩(8코어지만 보통 4개의 논리적 megacore로 취급)이다. TPU v5e 같은 추론 칩은 host당 tray가 1개가 아니라 2개지만 칩당 코어가 1개뿐이라, 8칩 = 8코어가 된다.[^4]

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/pcie.png" alt="PCIe로 host에 연결된 TPU tray" loading="lazy" />
</figure>

**PCIe bandwidth는 제한적이다:** HBM $\leftrightarrow$ VMEM 링크처럼 CPU $\leftrightarrow$ HBM PCIe 연결에도 정해진 bandwidth가 있어서, host 메모리에서 HBM으로 또는 그 반대로 얼마나 빨리 로드할 수 있는지가 제한된다. 예컨대 TPU v4의 PCIe bandwidth는 각 방향 16GB/s로, HBM보다 100배 가까이 느리다. host(CPU) RAM으로 데이터를 로드/오프로드할 수는 *있지만* 빠르지는 않다.

## TPU 네트워킹

**칩들은 Pod 안에서 ICI 네트워크를 통해 서로 연결된다.** 구세대(TPU v2와 TPU v3), 추론 칩(예: TPU v5e), Trillium(TPU v6e)에서 ICI("inter-chip interconnects")는 가장 가까운 4개의 이웃을 연결한다(edge 링크가 더해져 2D torus를 이룬다). TPU v4와 TPU v5p는 가장 가까운 6개의 이웃과 연결된다(3D torus를 이룬다). 이 연결은 host를 거치지 **않는**, 칩과 칩 사이의 직접 링크라는 점에 유의하라.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/ici-wraparound.png" alt="ICI wraparound 링크가 있는 torus 구조" class="img-small" loading="lazy" />
</figure>

torus 구조는 임의의 두 노드 사이의 최대 거리를 $N$에서 $N / 2$로 줄여 통신을 훨씬 빠르게 만든다. TPU에는 torus를 뫼비우스 띠 비슷한 topology로 감아 노드 간 평균 거리를 더 줄이는 "twisted torus" 구성도 있다.

**TPU pod(ICI로 연결된)는 정말 커질 수 있다:** 최대 pod 크기(**superpod**라고 부른다)는 TPU v4에서 `16x16x16`, TPU v5p에서 `16x20x28`이다. 이 거대한 pod들은 [optical wraparound link](https://arxiv.org/pdf/2208.10041)로 연결된 재구성 가능한 `4x4x4` 칩 큐브들로 구성되며[^5], 이를 재구성해 아주 큰 topology를 연결할 수 있다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/tpu-rack.png" alt="TPU 랙" loading="lazy" />
</figure>

더 작은 topology(예: `2x2x1`, `2x2x2`)도 요청할 수 있지만 wraparound는 없다. 대부분의 통신 시간이 통상 두 배가 되므로 중요한 단서 조항이다. 완전한 큐브의 배수(예: `4x4x4`나 `4x4x8`)는 optical switch가 wraparound를 제공한다.[^6]

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/subslices.png" alt="작은 TPU 서브슬라이스 topology" loading="lazy" />
</figure>

TPU v5e와 Trillium pod는 단일 `16x16` 2D torus로 구성되며, 크기 16인 축을 따라 wraparound가 있다(즉 `8x16`은 긴 축에 wraparound가 있다). TPU v5e와 v6e(Trillium)는 16x16 torus 너머로 확장할 수 없지만, pod끼리는 TPU host들을 서로 잇는 표준 데이터센터 네트워킹(DCN)으로 통신할 수 있다. 역시 더 작은 topology도 요청할 수 있으며 크기가 $<16$인 차원에는 wrap이 없다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/more-subslices.png" alt="TPU v5e 서브슬라이스 topology" loading="lazy" />
</figure>

**이 nearest-neighbor 연결이 TPU와 GPU의 핵심 차이다.** GPU는 TPU처럼 로컬 연결을 쓰는 대신, 모든 GPU 사이의 point-to-point 연결을 근사하는 스위치 계층으로 연결된다. 보통 한 노드 안의 GPU(H100은 8개, B200 NVL72는 최대 72개)는 직접 연결되고, 더 큰 topology에서는 GPU 사이에 O(log(N)) 홉이 필요하다. 한편으로 GPU는 적은 홉 수 안에 임의의 데이터를 보낼 수 있다. 다른 한편으로 TPU는 (NVLink 스위치가 비싸기 때문에) 극적으로 싸고, 배선이 단순하며, 디바이스당 링크 수와 디바이스당 bandwidth가 상수라서 훨씬 큰 topology로 확장할 수 있다. 더 자세한 내용은 [여기](/scaling-book/gpus/)를 읽어 보라.

**ICI는 DCN에 비해 매우 빠르지만, HBM bandwidth보다는 여전히 느리다.** 예컨대 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)는:

* 칩당 `2.8e12` bytes/s (2.8 TB/s)의 HBM bandwidth를 갖는다.
* 축당 `9e10` bytes/s (90 GB/s)의 ICI bandwidth를 가지며, 칩당 3개의 축이 있다.[^7]
* TPU당 `6.25e9` bytes/s (6.25 GB/s)의 DCN (egress) bandwidth를 갖는다(host당 1-2개의 NIC 경유).[^8]

그래서 모델을 여러 칩에 걸쳐 나눌 때는 더 느린 디바이스 간 통신이 MXU의 병목이 되지 않도록 조심해야 한다.

**Multi-slice 학습:** ICI로 연결된 TPU 집합을 **slice**라고 부른다. 서로 다른 slice는 DCN으로 연결될 수 있다. 예컨대 서로 다른 pod에 있는 slice들을 잇는 경우다. DCN은 ICI보다 훨씬 느린 연결이므로, 연산이 DCN에서 오는 데이터를 기다리는 일을 최대한 줄여야 한다. DCN은 host-to-host이므로, DCN을 통해 TPU에서 TPU로 버퍼를 옮기려면 먼저 PCIe로 host로 옮기고, 네트워크로 egress한 뒤, 대상 host의 네트워크로 ingress하고, 다시 PCIe로 HBM에 넣어야 한다.

## 핵심 요점

* TPU는 단순하며, 대부분의 경우 메모리(초고속), ICI를 통한 다른 칩들(꽤 빠름), DCN을 통한 데이터센터의 나머지(어느 정도 빠름)에 연결된 행렬 곱셈 유닛으로 생각할 수 있다.

* 통신은 속도 순으로 나열한 다음의 여러 네트워크 bandwidth에 의해 제한된다:
  * HBM bandwidth: TensorCore와 그에 딸린 HBM 사이.
  * ICI bandwidth: TPU 칩과 가장 가까운 4개 또는 6개 이웃 사이.
  * PCIe bandwidth: CPU host와 그에 딸린 칩 tray(들) 사이.
  * DCN bandwidth: 여러 CPU host 사이, 보통 ICI로 연결되지 않은 host들 사이.

* **slice 안에서 TPU는 ICI를 통해 nearest neighbor하고만 연결된다.** slice 안의 멀리 떨어진 칩 사이의 ICI 통신은 중간의 칩들을 먼저 거쳐야 한다.

* **weight 행렬은 MXU를 채우기 위해 양쪽 차원 모두 최소 크기 128**(TPU v6e에서는 256)**로 패딩되어야 한다** (실제로는 더 작은 축이 128로 패딩된다).

* **더 낮은 정밀도의 행렬 곱셈이 더 빠른 경향이 있다.** 이를 지원하는 세대에서 TPU는 int8이나 int4 OPs를 bfloat16 FLOPs보다 대략 2배/4배 빠르게 수행할 수 있다. VPU 연산은 여전히 fp32로 수행된다.

* TPU 연산 유닛에 병목이 생기지 않게 하려면 **각 채널을 오가는 통신량이 그 채널의 속도에 비례하도록** 해야 한다.

### TPU 스펙

우리 칩들의 구체적인 수치는 다음과 같다:

| 모델                                        | Pod 크기 | Host 크기 | 칩당 HBM 용량 | 칩당 HBM BW (bytes/s) | 칩당 FLOPs/s (bf16) | 칩당 FLOPs/s (int8) |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.2e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |
| <span class="nowrap-header">TPU7x</span>   | 4x4x576  |   2x2x1   |       192GB       |        7.4e12         |       2.30e15       |       4.61e15       |

Host 크기는 단일 host에 연결된 TPU들의 topology를 가리킨다(예: TPU v5e는 CPU host 하나가 4x2 topology의 TPU 8개에 연결된다). 최신 세대에 대한 자세한 내용은 [TPU7x 문서](https://docs.cloud.google.com/tpu/docs/tpu7x)를 참조하라. interconnect 수치는 다음과 같다:

| 모델        | link당 ICI BW (단방향, bytes/s) | link당 ICI BW (양방향, bytes/s) |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |             1.0e11             |           2.0e11            |
| **TPU v4p** |             4.5e10             |           9.0e10            |
| **TPU v5p** |             9.0e10             |           1.8e11            |
| **TPU v5e** |             4.5e10             |           9.0e10            |
| **TPU v6e** |             9.0e10             |           1.8e11            |
| **TPU7x**   |             9.0e10             |           1.8e11            |

단방향(one-way) bandwidth와 양방향(bidi) bandwidth를 둘 다 싣는 이유는, 단방향 bandwidth가 하드웨어에 더 충실한 수치이긴 하지만 완전한 ring이 관련된 수식에는 양방향 bandwidth가 더 자주 등장하기 때문이다.[^9]

PCIe bandwidth는 보통 TPU당 초당 약 `1.6e10` bytes(TPU v6e는 `3.2e10`)이고, DCN bandwidth는 보통 TPU당 초당 약 `6.25e9` bytes(TPU v6e와 TPU7x는 `12.5e9`, TPU v5e는 `3.125e9`)이다.

## 연습 문제

이 수치들이 좀 건조하긴 하지만, 이를 이용하면 모델 성능에 대한 기본적인 roofline 추정을 할 수 있다. 이게 왜 유용한지 설명하기 위해 몇 문제를 풀어 보자. 3부에서 더 많은 예제를 보게 될 것이다.

**문제 1 [LLM latency의 하한 잡기]:** 32개의 TPU v4p에 걸쳐 나뉜 bf16 200B 파라미터 모델에서 샘플링하고 싶다고 하자. 모든 파라미터를 HBM에서 systolic array로 로드하는 데 얼마나 걸리는가? *힌트: 위의 수치를 사용하라.*

<details>
<summary>정답 보기</summary>

**정답:** 32개 칩에 걸쳐 `sizeof(bf16) * 200e9 = 400e9` 바이트를 로드한다. 칩당으로는 12.5e9 바이트이고, 각 칩의 HBM bandwidth는 1.23e12이다. 따라서 로드에는 약 10ms가 걸린다.

꽤 멋진 결과다. *이것이 모델 샘플링 latency의 합리적인 하한*이기 때문이다. 각 샘플링 스텝은 모든 파라미터를 HBM에서 로드해야 하므로 10ms보다 빠를 수 없다. 실전에서는 작은 batch size일 때 이 값에 근접할 수 있다.

</details>

**문제 2 [TPU 세부사항]:** 완전한 TPU v5e pod를 생각하자. CPU host는 총 몇 개인가? TPU TensorCore는 몇 개인가? pod 전체의 총 FLOPs/s는? 총 HBM은? TPU v5p pod에 대해서도 같은 연습을 해 보라.

<details>
<summary>정답 보기</summary>

**정답:** TPU v5e는 pod가 `16x16`이고 host마다 4x2 slice이므로 host는 `16*16 / 8 = 32`개다. TPU v5e는 TPU마다 코어가 하나뿐이므로 TensorCore는 256개다. 총 FLOPs/s는 bfloat16 기준 `16*16*2e14 = 5.1e16`이다. 칩마다 16GB의 HBM이 있으므로 메모리는 `256 * 16 = 4TB`다.

완전한 TPU v5p pod는 칩이 `16x20x28`개이고 host마다 2x2x1이므로 host는 `(16*20*28) / (2*2) = 2,240`개다. TPU v5p는 TPU마다 TensorCore가 둘이므로 코어는 `8960 * 2 = 17,920`개다. 총 FLOPs/s는 bfloat16 기준 `8960 * 4.59e14 = 4.1e18`이다. 칩마다 96GB의 HBM이 있으므로 메모리는 `8960 * 96 = 860TB`다.

</details>

**문제 3 [PCIe operational intensity]:** 큰 weight 행렬 $A$($\text{bf16}[D, F]$ 타입)와 activation 배치 $x$($\text{bf16}[B, D]$ 타입)를 host DRAM에 저장할 수밖에 없는 상황에서, 이 둘의 행렬 곱셈을 하고 싶다고 하자. 단일 host에서 돌아가고, 거기에 붙은 TPU v6e 칩 하나를 쓴다. $B \ll D$이고 $F = 4D$라고 가정해도 된다(왜 이것이 합리적인 가정인지는 이후 장들에서 보게 된다). PCIe에 대해 FLOPs bound를 유지하기 위해 필요한 최소 batch size $B$는 얼마인가? PCIe bandwidth는 1.6e10 bytes/s로 가정한다.

<details>
<summary>정답 보기</summary>

**정답:** $2BDF$번의 부동소수점 연산을 수행해야 하고, 각 칩은 초당 `9.2e14`번의 부동소수점 연산을 수행할 수 있다. 따라서 여기에 $2BDF / 9.2e14$ 초가 필요하다. DRAM에서 $2DF + 2BD$ 바이트를 로드하고 $2BF$ 바이트를 다시 써야 한다. PCIe 전송 속도가 병목이므로, 데이터를 TPU로 오가게 하는 데 $2 \cdot (BD + DF + BF) / 1.6e10$ 초가 필요하다. 연산이 weight 로딩보다 오래 걸리기를 원하므로, weight 로딩 전부를 연산과 겹칠 수 있다고 가정하면 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.6e10$이 필요하다. $B \ll D$, $F = 4D$ 가정으로 단순화하면

$$
\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.6 \times 10^{10}}
$$

즉

$$
B > \frac{9.2 \times 10^{14}}{1.6 \times 10^{10}} \simeq 57{,}500
$$

이다.

</details>

**문제 4 [일반적인 matmul latency]:** weight 행렬 int8[16384, 4096]에 크기 int8[B, 4096]의 activation 행렬을 곱하고 싶다고 하자. B는 미지의 batch size다. 우선 TPU v5e 1개에서 시작하자.

1. 이 곱셈은 B의 함수로 얼마나 걸리는가? *힌트: 배열을 HBM에서 로드하는 데 걸리는 시간과 곱셈 자체에 걸리는 시간을 각각 계산해 보면 도움이 된다. 어느 쪽이 병목인가?*
2. 이 연산을 VMEM에서 돌리고 싶다면? B의 함수로 얼마나 걸리는가?

<details>
<summary>정답 보기</summary>

**정답:** (1) 수행해야 하는 연산 수는 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$이다. 따라서 $T_{\text{math}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ 초다. HBM에서 VMEM으로 $16384 \cdot 4096 + 4096 \cdot B$ 바이트를 로드하고, VMEM에서 HBM으로 $16384 \cdot B$ 바이트를 다시 써야 한다. 곧 $T_{\text{comms}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.2 \times 10^{11}$ 초다. 통신과 연산을 최대한 겹친다고 가정하면 전체 곱셈에는 대략 다음의 시간이 걸린다.

$$
\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}, \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}}\right\}
$$

$\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}} < \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$일 때, 즉 $B > 267$일 때 FLOPs-bound가 된다. $D$와 $F$의 영향을 전부 반영했기 때문에 [1장](/scaling-book/roofline/)에서 유도한 240보다 약간 큰 값이다.

(2) 대신 VMEM에서 로드한다면, MXU로의 VMEM bandwidth를 HBM $\leftrightarrow$ VMEM bandwidth의 22배로 잡자. 그러면 데이터 로딩의 분모가 8.2e11에서 1.80e13이 되고, $B > 11$을 얻는다. 실전에서는 VMEM bandwidth 전부를 weight 행렬 로딩에 쓸 수 없으므로, 실제로는 20에 가까울 것이다.

</details>

**문제 5 [ICI bandwidth]:** TPU v5e `4x4` slice가 있다고 하자. `bf16[8, 128, 8192]` 타입의 배열을 `TPU{0,0}`에서 `TPU{3, 3}`으로 보내고 싶다. TPU v5e의 홉당 latency는 $1\mu s$라고 하자.

1. 첫 바이트는 얼마 만에 목적지에 도착하는가?
2. 전체 전송에는 얼마나 걸리는가?

<details>
<summary>정답 보기</summary>

**정답:** TPU v5e는 2D 연결이다. `4x4` slice뿐이라(크기 16인 축이 없어서) wraparound 연결이 없다. 따라서 대상 칩이 데이터를 받을 수 있는 포트가 2개이고, 마찬가지로 소스 칩이 데이터를 보낼 수 있는 포트도 2개다. 전송해야 하는 데이터양은 `2 * 8 * 128 * 8192 = 1.7e7` 바이트다. 양쪽 포트에서 동시에 전송할 수 있으므로(즉 배열의 절반은 오른쪽으로, 절반은 아래로 보내면) 초당 `2 * 4.5e10 = 9e10` 바이트를 전송하고, (bandwidth bound라고 가정하면) 전체 배열을 통과시키는 데 약 `1.7e7 / 9e10 = 188us`가 걸린다. 칩이 16개 미만인 축에는 wraparound 링크가 없어서, `4x4` slice에서 칩 $(0, 0)$과 $(3, 3)$ 사이의 홉은 여섯 개다. 홉당 latency가 약 $1\mu s$이므로 첫 바이트는 약 `6us` 만에 도착하고, 전체 전송은 약 `188 + 6 = 194us`가 걸린다. 마지막 바이트 역시 소스를 떠난 뒤 여섯 홉을 거쳐야 하기 때문이다(일반적으로 latency 항과 bandwidth 항은 더해지는데, 여기서 latency는 작은 보정일 뿐이다).

</details>

**문제 6 [총정리, 어려움]:** 큰 행렬 **A**: `int8[128 * 1024, 128 * 1024]`가 TPU v5e 4x4 slice에 균등하게 sharding되어 있는데, 각 칩의 host DRAM에 오프로드되어 있다고 하자. 전체 배열을 TPU{0, 0}으로 복사한 뒤 벡터 `bf16[8, 128 * 1024]`와 곱하고 싶다. 얼마나 걸리는가? *힌트: 위의 수치를 사용하라.*

<details>
<summary>정답 보기</summary>

**정답:** 수행해야 할 작업부터 정리하자. 배열은 약 16GB다. 위 표에서 TPU v5e host는 4x2 topology이므로 4x4에는 host가 2개다. 배열이 균등하게 sharding되어 있으므로 각 host는 사실상 배열의 1/2, 즉 8GB 청크를 담고 있다. 이 청크들을 전부 TPU{0,0}으로 복사해야 하는데, 두 가지 선택지가 있다:

1. DCN으로 복사한 다음, sharding되지 않은 전체 배열을 PCIe로 HBM에 로드한다.
2. sharding된 배열을 각자의 대응 TPU에 로드한 다음 ICI로 gather를 수행하고, TPU{0,0}에서 matmul을 수행한다.

(2)가 낫다는 것은 분명하다. DCN은 ICI에 비해 느리고, 큰 배열은 몇 개의 링크(host 0의 8개)가 아니라 많은 PCIe 링크로 로드하는 편이 훨씬 낫다. 시스템 일부의 다이어그램이 아래에 있다. 위에서 설명했듯 TPU는 (host를 넘어서도) ICI로 이웃과 연결되고, 모든 TPU는 (PCIe로) 자기 host CPU에 연결되며, host들은 DCN으로 연결된다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/challenge-problem.png" alt="문제 6의 시스템 다이어그램" class="img-small" loading="lazy" />
  <figcaption>실제로는 칩마다 host로 가는 자기만의 PCIe 링크가 있지만, 그림을 명확하게 하기 위해 하나만 표시했다.</figcaption>
</figure>

이제 각 단계가 얼마나 걸릴지 계산해 보자:

1. **PCIe 로드**: 16GB의 청크를 16개의 PCIe 링크로 로드하며, 각 링크의 bandwidth는 `1.6e10` bytes/s다. 따라서 약 63ms가 걸린다.

2. **ICI 복사:** 이제 각 TPU는 배열의 16GB / 16 = 1GB를 갖고 있다. ICI bandwidth는 링크당 *양방향* 9e10 bytes/s인데, 위 다이어그램을 보면 이 topology에서 TPU{0,0}은 TPU v5e의 ICI 링크 4개 중 2개만 사용한다. TPU{0,0}은 총 15GB를 2개의 축을 따라 링크당 `4.5e10` bytes/s로 받아야 하므로, 시간의 하한은 `15e9 / (4.5e10 * 2) = 167ms`다. 실전에서는 로드가 매우 불균등해서 아마 달성하기 어렵겠지만, 2배 이내일 것이다. 3장에서 보겠지만 full AllGather를 수행해도 대략 `16e9 / (4.5e10 * 2)`가 걸리므로 최적에 가깝다.

3. **HBM $\rightarrow$ MXU 로드:** 마지막 matmul을 수행하려면 이 16e9 바이트에 더해 bf16[8, 128 \* 1024] 배열(추가 2MB이므로 무시 가능)을 HBM bandwidth로 MXU에 로드해야 하는데, 여기에 `16e9 / 8.2e11 = 20ms`가 걸린다.

4. **FLOPs:** 총 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}$$ FLOPs를 수행하며, `1.97e14` bf16 FLOPs/s를 수행할 수 있으므로 1.4ms가 걸린다.

총 시간의 상한은 이 시간들의 합이지만, TPU는 보통 이 연산들을 겹칠 수 있으므로 가장 느린 조각이 병목이 되는 pipelining 문제로 생각하면 된다. 그렇다고 가정하면 답은 최소 167ms이고, 겹침이 불완전하면 200ms에 가까울 것이다.

</details>

<div class="takeaway">

**2부는 여기까지!** partitioning과 TPU 간 통신을 다루는 3부는 [여기](/scaling-book/sharding/)에서 볼 수 있다.

</div>

## 부록

### 부록 A: TPU 내부 구조 더 보기

여기서는 TPU의 내부 동작을 좀 더 깊이 들여다본다. 별도의 언급이 없으면 TPU v5p 기준의 스펙을 제시한다.

### VPU

VPU는 TPU의 벡터 산술 코어다. VPU는 vadd(벡터 덧셈)나 vmax(원소별 max) 같은 원소별 산술 연산을 수행하는 2차원 SIMD 벡터 머신(**VPU**)과, VPU와 MXU를 위한 데이터를 담는 벡터 레지스터 집합인 **VREG**로 구성된다.

**VREG:** TPU v5p 코어마다 64개의 32비트 VREG가 있고(TPU v4는 32개), 코어당 총 약 `64 * 8 * 128 * 4 = 256kB`의 VREG 메모리를 갖는다(코어가 둘이므로 칩 전체로는 이것의 2배). TPU v5p는 사이클마다 VMEM에서 레지스터 3개를 로드하고, VMEM에 레지스터 1개를 쓸 수 있다.

**VPU:** VPU는 `(8, 128)` shape의 2D 벡터 산술 유닛인데, 128 차원을 lane 축이라 부르고 8 차원을 sublane 축이라 부른다. v5에서 각 (lane, sublane) 쌍은 서로 독립적인 표준 부동소수점 ALU 4개를 담고 있다. VPU는 대부분의 산술 명령(vadd 즉 vector add 같은)을 각 ALU에서 한 사이클에 실행하고 latency는 2사이클이므로, 예컨대 v5에서는 사이클마다 VREG에서 가져온 f32 값 4쌍을 더할 수 있다. 전형적인 VPU 명령은 `{v2 = vadd.8x128.f32 v0, v1}`처럼 생겼는데, v0과 v1이 입력 VREG이고 v2가 출력 VREG다.

모든 lane과 sublane은 순수 SIMD 방식으로 사이클마다 같은 프로그램을 실행하지만, ALU마다 다른 연산을 수행할 수는 있다. 그래서 예컨대 vadd 1개와 vsub 1개를 한 사이클에 처리할 수 있으며, 각각은 두 개의 전체 VREG에 대해 연산하고 결과를 세 번째 VREG에 쓴다.

**Pop Quiz [VPU throughput 계산하기]:** 위 정보를 이용해 TPU v5p가 초당 몇 개의 벡터 FLOPs를 수행할 수 있는지 계산하라. TPU v5p의 클럭 속도는 약 1.75GHz다.

<details>
<summary>정답 보기</summary>

*정답*: 사이클마다 각 코어는 `8 * 128`개의 ALU에서 벡터 명령 4개를 실행할 수 있다. 코어당 `8 * 128 * 4` FLOPs/cycle, 곧 `8 * 128 * 4 * 1.75e9 = 7e12 FLOPs/s`다. 코어당 약 `2e14`인 MXU FLOPs/s보다 얼마나 작은지 보라(대략 30배).

</details>

**Reduction:** 일반적으로 sublane 차원을 가로지르는 통신이나 reduction이 lane 차원을 가로지르는 것보다 쉽다. 예컨대 VPU는 크기 8인 축을 따라 약 한 사이클 만에 roll할 수 있는 intra-lane shuffle 연산을 지원한다. 이를 이용해 sublane 차원의 효율적인 reduction을 수행할 수 있다(4, 2, 1씩 shuffle하며 세 번의 원소별 합을 하면 된다).

lane을 가로지르는 reduction은 훨씬 어렵고, XLU 즉 "cross lane unit"이라는 별도의 하드웨어 유닛을 거치는데, 느리고 꽤 비싸다.

**GPU와의 비교:** NVIDIA GPU에 익숙한 독자라면, VPU의 각 ALU는 CUDA core에 해당하고, VPU lane 하나는 "Warp Scheduler" — 즉 SIMD 산술을 수행하는 (보통 32개의) CUDA Core 집합 — 에 해당한다. lane 안에서의 reduction은 꽤 쉽지만, lane을 가로질러야 한다면 최소한 VMEM/XLU/SMEM을 거쳐야 해서 훨씬 느리다. 자세한 내용은 [GPU 장](/scaling-book/gpus/)을 보라.

### Scalar Core

scalar core는 TPU의 제어 유닛이다. 모든 명령을 fetch하고 dispatch하며, HBM에서 VMEM으로의 전송을 실행하고, 스칼라 메타데이터 작업을 하도록 프로그래밍할 수 있다. scalar core가 싱글 스레드이기 때문에 생기는 부작용 하나는, TPU의 각 코어가 사이클당 DMA 요청을 하나만 만들 수 있다는 것이다.

맥락을 짚어 보면, scalar core 하나가 VPU(4096개의 ALU로 구성), MXU 4개, XLU 2개, 그리고 여러 DMA 엔진을 제어한다. 연산 유닛당 제어가 이렇게 극도로 치우친 구조는 하드웨어 효율의 원천이지만, 데이터 의존적인 vectorization을 흥미로운 방식으로 할 수 있는 여지를 제한하기도 한다.

### 부록 B: systolic array는 어떻게 동작하는가?

TPU MXU의 핵심에는 `128x128` systolic array가 있다(TPU v6e는 `256x256`). 완전히 포화되면 systolic array는 8 클럭 사이클마다 한 번의 `bf16[8,128] @ bf16[128,128] -> f32[8,128]`[^10] 곱셈을 수행할 수 있다.

* 핵심만 보면 systolic array는 각각 곱셈과 덧셈 연산을 수행할 수 있는 ALU들의 2D `128x128`(`=16,384`) 그리드다.
* weight(**W**, `128x128` 입력)는 위에서 아래로 전달되고(RHS라 부른다), 입력(**X**, `8x128` 입력)은 왼쪽에서 들어온다(LHS라 부른다).

다음은 weight 집합(파란색)과 activation 집합(초록색)을 곱하는 단순화된 애니메이션이다. weight(RHS)가 먼저 대각선 방향으로 부분 로드되고, 그다음 activation이 역시 대각선 방향으로 공급되는 것을 볼 수 있다. 아래의 각 프레임에서 겹쳐진 초록색과 파란색 유닛을 전부 곱하고, 위에서 전달된 잔여값이 있으면 그 결과와 합한 뒤, 결과를 차례로 한 유닛 아래로 전달한다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/systolic-array.gif" alt="systolic array 곱셈 애니메이션" loading="lazy" />
</figure>

계산 결과가 스트리밍되어 나오는 것까지 보여주는, 더 일반적인 버전의 애니메이션:

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/systolic-array2.gif" alt="출력 스트리밍을 포함한 systolic array 애니메이션" class="img-small" loading="lazy" />
</figure>

여러 RHS·LHS 배열에 걸쳐 이를 pipeline하는 방식을 보여주는 다이어그램:

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/systolic-array-pipelining.png" alt="systolic array pipelining 다이어그램" loading="lazy" />
</figure>

weight(RHS)와 activation(LHS)이 로드되는 동안 초기 pipeline bubble이 생긴다. 그 초기 bubble 이후에는 추가 bubble 없이 새 입력과 weight를 로드할 수 있다.

다음은 bf16[2, 3] x bf16[3, 3] 행렬 곱셈의 (조악한) 애니메이션인데, 2x3 weight 행렬을 batch 1에 크기 3인 입력 activation과 곱하는 matmul이라고 생각할 수 있다. 앞의 슬라이드들과 비교해 회전되어 있고 입력이 아래가 아니라 오른쪽으로 흘러나가지만, 구조는 대략 볼 수 있다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/systolic-array-bad.gif" alt="bf16[2,3] x bf16[3,3] 곱셈 애니메이션" class="img-small" loading="lazy" />
</figure>

이를 효율적으로 pipeline하면 너무 큰 pipeline bubble 없이 큰 행렬들을 곱할 수 있다. 그렇긴 해도 행렬의 shape이 MXU의 변 크기(일반적으로 128x128)보다 큰 것이 중요하다. 일부 TPU는 (TPU v3부터) MXU를 여러 개 갖는데 TPU v3는 2개, TPU v4/5는 4개이므로, tiling 차원이 128 * MXU 수보다 크도록 해야 한다. [여기](https://www.youtube.com/watch?v=sJltBQ4MOHA)에 이를 잘 보여주는 애니메이션이 있다.

Trillium(TPU v6e)은 `256x256` systolic array를 가져서 사이클당 4배의 FLOPs를 수행할 수 있다. MXU를 온전히 활용하려면 텐서의 차원도 두 배로 커야 한다는 말이기도 하다.

[이 블로그 포스트](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)에는 고정된 weight 행렬에 대한 systolic array 곱셈을 보여주는 또 하나의 훌륭한 애니메이션이 있다.

[^1]: TPU v6e(Trillium)는 256x256 MXU를 갖고, 이전 세대들은 모두 128x128을 쓴다.
[^2]: TPU, 특히 그 systolic array가 이토록 강력한 하드웨어 가속기인 이유는, 행렬 곱셈이 $O(n^2)$ 바이트에 대해 $O(n^3)$의 연산을 쓰는 몇 안 되는 알고리즘 중 하나이기 때문이다. 덕분에 평범한 ALU가 메모리 bandwidth가 아니라 연산에 의해 병목이 되기가 아주 쉽다.
[^3]: VMEM prefetching이라는 말을 종종 하는데, matmul에 쓸 weight를 미리 VMEM에 로드해서 로딩 비용을 가리는 것을 뜻한다. 예컨대 일반적인 Transformer에서는 attention이 수행되는 동안 큰 feed-forward weight를 VMEM에 로드해 둘 수 있고, memory bandwidth bound인 상황이라면 weight 로드 비용을 숨길 수 있다. 이렇게 하려면 weight가 충분히 작거나 충분히 sharding되어 있어서, 한 레이어가 여유 공간을 남기고 VMEM에 들어가야 한다.
[^4]: Cloud TPU VM에서는 각 tray가 별도 VM의 일부로 노출되므로, 다시 4개의 코어가 보이게 된다.
[^5]: optical switch는 같은 ICI bandwidth를 가진 재구성 가능한 연결일 뿐이다. wraparound 링크를 유지하면서 큐브들을 연결할 수 있게 해 준다.
[^6]: `2x2x4`에는 wraparound가 전혀 없다는 점에 유의하라. wraparound는 완전한 큐브에서만 제공되는 optical switch가 만들어 주기 때문이다. 반면 TPU v5e 8x16은 재구성 가능한 optical 네트워킹을 쓰지 않으므로 긴 축에 wraparound가 *있다*.
[^7]: 위 페이지에는 100 GB/s의 bandwidth로 적혀 있어 여기 수치와 약간 다르다. TPU ICI 링크는 수행하는 연산에 따라 bandwidth가 조금씩 다르다. 이 문서의 수치는 대체로 걱정 없이 사용해도 된다.
[^8]: TPU v6e와 TPU7x는 12.5e9 bytes/s, v5e는 3.125e9 bytes/s다.
[^9]: bidi(양방향) bandwidth란 하나의 링크를 따라 양쪽 방향으로 보낼 수 있는 총 바이트 수, 또는 똑같은 말로, 두 링크를 모두 효율적으로 쓸 수 있다고 가정할 때 특정 축을 따라 단일 TPU에서 나가는 총 바이트 수를 뜻한다. 이는 ring이 제대로 동작할 때, 즉 해당 축에 wraparound 연결이 있을 때 성립한다. 추론 칩에서는 완전한 크기 16의 축이 있을 때, 학습 칩(v*p)에서는 축이 4의 배수일 때 그렇다. 양방향 통신이 관련된 계산에 자주 등장하기 때문에 양방향 bandwidth를 선호한다.
[^10]: 이 표기가 낯설다면: bfloat16 원소를 가진 `8x128` 행렬과 bfloat16 원소를 가진 `128x128` 행렬을 곱해, 결과를 float32 원소를 가진 `8x128` 행렬에 저장한다는 뜻이다.
