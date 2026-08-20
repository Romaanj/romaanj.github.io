---
part: 8
title: "TPU에서 LLaMA 3 서빙하기"
title_en: "Serving LLaMA 3-70B on TPUs"
original: "https://jax-ml.github.io/scaling-book/applied-inference/"
summary: "LLaMA 3-70B를 TPU v5e에서 서빙하는 데 필요한 모든 것 — KV cache 크기, batch size, topology 선택, sharding — 을 back-of-the-envelope 계산으로 직접 따라가 본다. latency와 throughput의 극적인 트레이드오프를 숫자로 확인한다."
date: 2026-08-20
published: true
---

> LLaMA 3-70B 모델을 TPU v5e에서 서빙하는 방법을 자세히 들여다보자. 서로 다른 모델을 roofline 기준으로 서빙하면 비용이 얼마나 드는가? KV cache는 얼마나 큰가? 어떤 batch size를 써야 하는가? 추론 중에 파라미터와 activation은 어떻게 sharding되는가? 프로덕션에서의 latency와 throughput을 어림(back-of-the-envelope) 계산으로 추정해 보자.

*이 장에서는 LLaMA-3를 서빙하는 데 무엇이 필요하고 얼마나 효율적으로 할 수 있는지 살펴본다. 앞의 "응용" 장과 마찬가지로, 정답을 찾아보기 전에 펜과 종이로 직접 답을 구해 보자!*

## LLaMA 서빙 스토리

LLaMA 3-70B가 어떤 모델인지 다시 떠올려 보자([6장](/scaling-book/applied-training/) 참조):

| **하이퍼파라미터**          | **값**    |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

간단한 질문에서 시작하자: **어떤 하드웨어에서 서빙해야 하는가?** 답은 기본적으로, FLOPs / 달러가 가장 싼 하드웨어다.[^1] 그래서 보통 현재의 전용 추론 칩인 TPU v5e에서 서빙하려 한다(비용은 2025년 2월 기준 [Google Cloud 가격](https://cloud.google.com/tpu/pricing)):

| **TPU 종류** | **bfloat16 FLOPs/s** | **Google Cloud USD / 시간** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

TPU v5e 하나에는 HBM이 16GB뿐이라 모델을 꽤 공격적으로 sharding해야 한다. 중요할 만한 기본 수치 몇 가지부터 생각해 보자:

**질문:** LLaMA 3-70B의 KV cache는 토큰당 얼마나 큰가? *int8로 저장한다고 가정해도 된다. 이 값이 주어진 topology에서 batch size를 얼마나 키울 수 있는지를 결정한다.*

<details>
<summary>충분히 생각해 봤다면 클릭!</summary>

LLaMA 3-70B는 KV head가 8개이므로 토큰당 크기는 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`이다.

**이게 얼마나 큰지 눈여겨보라!** 시퀀스 길이가 (흔히 그렇듯) 32k 토큰이면 시퀀스당 `160e3 * 32,768 = 5.3GB`를 쓴다. BS=240이면 1.3TB다! TPU v5e는 개당 16GB뿐이므로 이만큼의 메모리를 담는 데만 약 `(70e9 + 1.3e12) / 16e9 = 86`개의 TPU v5e 칩이 필요하다. 모델 파라미터 70GB에 비해 얼마나 큰지도 눈여겨보라.

</details>

**질문:** L3 70B를 batch size 32, 시퀀스 길이 8192로, 전부(파라미터와 KV 모두) int8로 서빙하고 싶다고 하자. 총 메모리를 얼마나 쓰게 되는가? 서빙할 수 있는 가장 작은 slice는 무엇인가?

<details>
<summary>정답 보기</summary>

KV가 int8에서 `160e3` 바이트이므로 총 KV 메모리는 `160e3 * 8192 * 32 = 41.9e9` 바이트다. 파라미터는 파라미터당 1바이트이므로 `70e9` 바이트다. 따라서 총 메모리 사용량은 `41.9e9 + 70e9 = 112GB`다.

쓸 수 있는 가장 작은 slice는 `112e9 / 16e9 = 7` TPU, 즉 (짝수 크기로 반올림하면) TPU v5e `4x2`다. 꽤 빠듯하게 들어가는 수준이라 다른 오버헤드까지 감안하면 다 안 들어갈 수도 있으므로, 최소 `4x4`가 필요할 수도 있다(아니면 batch size를 낮추거나).

</details>

**질문:** 이 batch size와 quantization으로 TPU v5e `4x2`에서 서빙하면 decode step당 latency는 대략 얼마로 예상되는가? throughput(tokens / sec / chip)은? `4x4`라면? *FLOPs는 bfloat16으로 수행하고 모든 것이 완전히 sharding되어 있다고 가정한다.*

<details>
<summary>정답 보기</summary>

이전 장의 공식을 그대로 가져오면 된다:

$$
\begin{align*}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align*}
$$

여기서는 파라미터가 int8이고 FLOPs는 bfloat16이므로 임계 batch size는 약 120이다. 오른쪽 항의 max를 직접 계산할 수도 있지만, 그건 이미 여러 번 해 본 계산이다. **matmul과 FLOPs 양쪽 모두 memory-bound 영역 깊숙이 들어와 있다.**

그러면 메모리 bandwidth만 엄밀히 봤을 때 step time은 기본적으로 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.2e11) = 17ms`다. **따라서 이론적인 step time은 약 17ms다.** throughput은 `32 / .017 = 1882 tokens / sec`, 칩당으로는 `1882 / 8 = 235 tokens / sec / chip`이 된다.

한 가지 확인해 둘 것은 matmul이 ICI bound일 가능성이다. 여기서는 2개 axis를 여기에 쓸 수 있으므로, 이론상 ICI bound가 되는 조건은 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$이다. 그러니 문제없다!

`4x4`에서 돌려도 ICI 측면에서는 여전히 괜찮으므로 latency는 `17 / 2 = 8.5ms`로 떨어지지만, 칩당 throughput은 그대로다.

</details>

### throughput 중심으로 생각하기

잠시 순수하게 throughput만 생각해 보자. throughput을 최적화할 때는 compute-bound가 되기를, 곧 TPU MXU 용량을 거의 전부 활용하는 상태에 가까워지기를 원한다. 보통은 batch size를 최대한 키워서 가능한 한 많은 일을 한꺼번에 하고 있어야 한다.

**질문:** TPU v5e에서 bfloat16 weight와 activation을 쓸 때, matmul이 compute-bound가 되려면 batch size가 얼마나 커야 하는가? weight는 int8로 하되 FLOPs를 bfloat16으로 수행하면? int8 weight에 int8 FLOPs라면?

<details>
<summary>정답 보기</summary>

7장에서 논의했듯이 $B \ll D, F$인 bfloat16 matmul에서는

$$
\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}
$$

이다. weight가 int8이면 분모에서 2배가 사라지므로 $2BDF / DF = 2B > 240$, 즉 $B > 120$이 되어 임계 batch size가 이전의 절반이 된다. 우리에게 정말 유리하다! int8 weight에 int8 FLOPs를 쓰면 TPU FLOPs/s로 int8 값을 써야 하는데, 이 값은 bfloat16의 1.97e14에서 3.94e14로 거의 두 배가 된다. 그러면 다시 원점인 약 $B > 240$으로 돌아온다.

int8 weight + bfloat16 FLOPs 조합은 꽤 흔하다. 파라미터를 손실 없이 quantize하는 것이 저정밀 산술 연산을 수행하는 것보다 쉬운 경우가 많기 때문이다.

</details>

**질문:** 8k 컨텍스트에서 bfloat16, int8, int4(KV와 파라미터 모두)로 LLaMA 3-70B를 서빙할 수 있는 가장 작은 TPU v5e topology는 무엇인가? *이 문제에서는 KV cache가 무시할 만큼 작다고 생각해도 된다.*

<details>
<summary>정답 보기</summary>

쉽다! 아주 작은 batch size라도 괜찮다면 유일한 제약은 파라미터 메모리를 HBM에 담는 것이다. `ceil(num_params * sizeof(dtype) / HBM per TPU)`, 다시 말해 `ceil(70e9 * sizeof(dtype) / 16e9)`를 가장 가까운 합리적인 topology(2의 배수)로 올림하면 된다:

| dtype | 파라미터 크기 | 토큰당 KV 크기 (bytes) | 최소 TPU v5e 수 | 실제 최소 slice | KV cache용 잔여 HBM | 8k 기준 KV cache 수 |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 chips  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             29              |         43         |

꽤 멋지다! 원한다면 LLaMA 70B를 TPU v5e 2x2에 넣을 수 있다는 뜻이니까. 다만 KV cache 수가 아주 작다는 점이 눈에 띌 것이다. 그게 곧 우리의 batch size다! FLOPs 활용률이 끔찍하게 낮을 것이라는 뜻이다. batch size를 240까지 끌어올릴 수 있다면 우리는 기꺼이 더 큰 topology를 쓸 것이다.

</details>

**질문:** 이 topology들에 들어가는 가장 큰 batch size를 쓴다고 하자. 각 generate step에서 어떤 latency를 기대할 수 있는가?

<details>
<summary>정답 보기</summary>

이것도 쉽다. HBM을 꽉 채우도록 batch size를 골랐기 때문이다! TPU v5e 한 대 분량의 바이트를 MXU로 로드하는 데 얼마나 걸리는지의 문제일 뿐이다. 그냥 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`, 즉 **step당 19ms**다. 생성 길이의 중앙값이 512 토큰이라고 가정하면 decode 한 번에 약 9초다. batch size를 줄이면 latency가 조금은 나아질 수 있다는 점도 알아두자. 예를 들어 int4에서 모델 파라미터만 따진다면 HBM이 더 이상 꽉 차 있지 않으므로 최소 latency는 약 10ms / step이다.

</details>

<div class="takeaway">

**요점(Takeaway):** decode latency의 하한은 언제나 "모델 파라미터 전부를 HBM에서 MXU로 로드하는 데 얼마나 걸리는가"로 잡힌다. KV cache가 작을 때는 각 layer가 weight를 청크 단위로 로드했다가 버리는 것으로 생각해도 된다. 큰 batch size나 대량의 디바이스 간 통신을 쓰지 않는 한 대체로 합리적인 하한이다(1.5배 이내). batch size가 커지면 KV cache 로딩이 파라미터를 압도하므로 KV cache 로딩도 모델링해야 한다.

</div>

마찬가지로 FLOPs-bound 영역(예: 학습이나 큰 배치의 추론)에서는 통신이 없다고 가정한 하한 $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$를 쓰면 된다.

**질문:** 각각의 경우 칩당 throughput은 얼마가 되는가(queries / chip 기준)? *decode 길이의 중앙값은 512 토큰이라고 가정해도 된다.*

<details>
<summary>정답 보기</summary>

토큰당 비용과 정확히 상관되는 값이므로 중요한 질문이다.

decode 길이 중앙값 가정을 쓰면 throughput은 그냥 $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approx 43 / (0.019 * 512 * N)$$이다. 대략 $$(4.42 / N)$$ QPS가 나오므로, $$N$$을 대입하면:

|  dtype   | QPS / chip |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

forward pass의 작업 메모리(activation과 attention에 할당되는 메모리)를 완전히 무시했으므로 상당히 낙관적인 수치다. Flash Attention이라면 터무니없는 가정은 아니지만 현실적이지도 않다. 실제 수치는 이것의 절반 정도일 가능성이 크다. 절대적인 최대 throughput을 원한다면 아마 칩 수를 두 배 이상으로 늘리고 batch size도 크게 키워야 할 것이다.

</details>

**질문:** 위 예시 각각에서 topology를 두 배로 늘리면 peak throughput은 어떻게 달라지는가?

<details>
<summary>정답 보기</summary>

bfloat16으로 4x8 slice를 쓰면 KV cache용으로 372GB가 남아 batch size를 140까지 키울 수 있다. 그러면 step time은 그대로이므로 throughput은 `14.39 / num_chips`가 되어:

|       dtype       | QPS / chip |
| :---------------: | :--------: |
| bfloat16 (on 4x8) |    0.44    |
|   int8 (on 4x4)   |    0.90    |
|   int4 (on 2x4)   |    1.80    |

더 키우면 이득이 훨씬 더 커진다! 여기서 큰 교훈은, KV cache 크기에 제약을 받는 상황이라면 모든 경우에 **가장 작은 topology가 가장 성능 좋은 topology는 아니라는 것**이다.

</details>

**질문:** 이제 sharding 문제를 파고들어 보자. TPU v5e 4x8에서 bfloat16으로 서빙하고 싶다고 하자. generation 중에 TPU v5e 4x8에서 모델에 어떤 sharding을 쓰게 되는가? communication-bound를 피할 수 있는가?

<details>
<summary>정답 보기</summary>

이전 장에서 논의했듯이 generation 중의 sharding 선택지는 사실상 하나뿐이다: model parallelism. communication-bound가 되기 전까지 얼마나 할 수 있을까? 역시 이전 장에서 봤듯이, 모델은 대략 다음 조건에서 communication-bound가 된다:

$$
Y > \frac{F \cdot M_Y}{2200}
$$

LLaMA 3-70B는 `F = 28,672`이므로, model sharding에 2개 axis를 쓰면 대략 $$Y = 28672 \cdot 2 / 2200 = 26$$이 나온다. 즉 일반적으로 약 16개 칩까지는 communication-bound 없이 확장할 수 있고 `4x4`는 쓸 수 있지만 `4x8`은 안 된다. 게다가 연산을 완벽히 겹치지는 못하므로 이 추정치조차 지나치게 낙관적이다.

**요점: 순수 model parallelism으로는 실제로 4x8에서 서빙할 수 없다.** 여기서 할 수 있는 최선은 4x2이고, _어쩌면_ 4x4까지다.

하지만 앞서 논의했듯이 batch size가 작을 때는 모델이 FLOPs-bound가 아니라 memory-bandwidth-bound이므로, 흔히 throughput을 크게 해치지 않으면서 model parallelism을 더 할 수 있다. 앞에서 이 값이 대략 $Y=F / (8\cdot B)$라고 했으므로, batch size 64라면 이론상 ICI-bound가 되기 전까지 `Y = 28,672 / (8 * 64) = 56`-way model parallelism까지 갈 수 있다. 검산을 위해 단일 matmul의 $T_\text{ici comms}$, $T_\text{hbm comms}$, $T_\text{math}$를 보자. 명확하게:

$$
\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}
$$

`4x8`에서는 $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`, $T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.2e11) = 18us`, $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`가 되어, 이론상 여전히 HBM bandwidth bound다. 아주 좋다! *다만 `4x4`에서 `4x8`로 키우는 것은 throughput 관점에서는 아마 별 도움이 안 되고, 대신 latency를 줄여 준다!*

int8과 int4 config를 보면, 그쪽은 순수 model parallelism으로 _할 수 있다_. 이제 quantization이 더 빠른 FLOPs를 넘어 실질적인 이점을 주는 지점에 도달한 것이다: comms-bound가 되기 전에 더 큰 batch size를 쓸 수 있게 해 준다. **이 이야기의 결말은, 4x8에서 peak throughput은 달성할 수 없지만 int8과 int4 config라면 순수 model parallelism이 가능하다는 것이다.**

</details>

<div class="takeaway">

**팁(Tip):** 유용한 model parallelism의 최대치는 $$d_{ff}$$와 모델을 sharding하는 axis 수에 달려 있다. 최댓값은 모델 크기에 따라 보통 8에서 32 사이다. 이 한계를 넘어 확장해서 throughput을 일부 희생하고 latency를 개선할 수도 있다.

</div>

### prefill은 어떤가?

지금까지 prefill은 훨씬 단순하다는 이유로 대체로 무시해 왔다. 이제 몇 가지 개념을 엮어 end-to-end 그림을 생각해 보자.

**질문:** prefill 중 40%의 FLOPs 활용률을 달성한다고 하자. 길이 8192의 prefill은 16개의 TPU v5e 칩에서 얼마나 걸리는가?

<details>
<summary>정답 보기</summary>

8k 토큰이면 확실히 compute-bound이므로 FLOPs만 따지면 된다. 모델 파라미터가 `70e9`개이므로 forward pass 한 번에 `2 * 70e9 * B` FLOPs를 쓴다. 40% MFU(FLOPs 활용률)를 가정하면 실행 시간은 약 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`다. 지금까지 봐 온 수치들과 비교하면 사실 꽤 긴 시간이다!

</details>

**질문:** prefill 길이의 중앙값이 8192 토큰, decode 길이의 중앙값이 4096 토큰이라고 하자. generate batch size는 32다. 평균적으로 step당 몇 개의 시퀀스가 decoding을 끝내는가? 평균적으로 step마다 몇 개의 토큰이 KV cache에서 evict되는가?

<details>
<summary>정답 보기</summary>

꽤 단순한 문제다. decode 길이 중앙값이 4096 토큰이므로 시퀀스는 대략 1 / 4096 토큰마다 하나씩 끝난다. batch size가 32이므로 step당 `32 / 4096`개의 시퀀스가 evict된다. KV cache 길이가 대략 `8192 + 4096`이므로 step당 evict되는 토큰은 `32 * (8192 + 4096) / 4096 = 96`개다. 일반 공식은 $B * (P + G) / G$이며, 여기서 $P$와 $G$는 각각 prefill과 generate 길이다.

</details>

**질문:** prefill 길이 중앙값 8192, decode 길이 중앙값 512로 disaggregated serving을 한다고 하자. 위에서 bfloat16으로 계산한 prefill·generate latency를 가정한다. 양쪽을 모두 완전히 포화 상태로 유지하려면 prefill:generate 서버 비율이 얼마나 필요한가?

<details>
<summary>정답 보기</summary>

꽤 재미있는 문제다. prefill 서버 수를 $P$, generate 서버 수를 $G$라고 하자. 일반적으로 `P / prefill_latency`의 속도로 시퀀스를 밀어 넣고 `B * G / (generate_latency * median_decode_length)`의 속도로 소비하는 pipeline 문제다. 앞에서 prefill step당 `910ms`, batch size 43(그냥 32라고 치자)에서 decode step당 `19ms`를 계산했다. 따라서 `P / 0.91 = 32 * G / (0.019 * 512)`, 즉 `P = 3G`가 필요하다. generation 서버보다 prefill 서버가 약 3배 더 필요하다!

</details>

## Latency-Throughput 트레이드오프 시각화하기

LLaMA 70B를 조금만 더 붙들고, generation 중 batch size에 따른 latency와 throughput을 실제로 살펴보자. 이전 장에서 PaLM 모델로 보였듯이 이렇게 하면 throughput/latency의 Pareto frontier가 만들어진다. MLP 블록에서 compute-bound를 유지하면서 쓸 수 있는 합리적인 상한이 16-way tensor parallelism이므로 그렇게 가정한다. 여기서는 TPU v5e 4x4 topology를 쓴다. **슬라이더로 시퀀스 길이를 조절하면 KV cache가 커질 때의 효과를 볼 수 있다.**

<div class="l-page">
  <iframe src="https://jax-ml.github.io/scaling-book/assets/plotly/pareto.html" frameborder="0" scrolling="no" height="400px" width="100%"></iframe>
</div>

* **비용과 latency 사이의 트레이드오프가 얼마나 극적인지 보라.** 토큰당 latency를 두 배로 치르는 대가로 토큰당 비용을 대략 100배 줄일 수 있다. 또한 latency는 작은 batch size에서의 5.5ms부터 아주 큰 배치에서의 20ms까지 어디든 될 수 있다.
* 2k 컨텍스트에서 BS 120 roofline에 도달하면 throughput이 사실상 약 1 token / ms / chip에서 정체되는 것에 주목하라(int8 weight에 bf16 FLOPs를 쓰므로 여기서는 120이다). 하지만 시퀀스 길이가 길어지면 이 batch size를 더 이상 메모리에 담을 수 없어서 완전 포화 지점에는 결코 도달하지 못한다.
* 같은 throughput이라도 큰 batch size에서는 latency가 얼마나 더 높은지 주목하라. (파라미터 로딩 대신) KV 로딩이 지배적이 되기 때문이다.

비용과 latency의 원천을 파라미터 로딩 시간, KV 로딩 시간, FLOPs 시간으로 분해해 보면 더 잘 이해된다. 빨간 영역은 MLP 블록에서 compute-bound가 될 것으로 기대되는 구간이다.

<div class="l-page">
  <iframe src="https://jax-ml.github.io/scaling-book/assets/plotly/latency_breakdown_log.html" frameborder="0" scrolling="no" height="400px" width="100%"></iframe>
</div>

많은 것을 말해 주는 그림이다. 처음에는 파라미터 로딩이 latency의 대부분을 차지하다가, batch size가 충분히 커지면 FLOPs와 KV 로딩이 더 중요해진다. 특히 시퀀스 길이가 2048을 넘는 모든 경우에 FLOPs보다 KV cache 로딩에 더 많은 시간을 쓴다! **batch size를 키워 하드웨어 활용률을 높일 수는 있지만, 긴 컨텍스트 길이에서는 KV 로딩이 항상 전체 step time을 지배한다.**

<div class="takeaway">

**요점(Takeaway):** LLaMA 3-70B는 이 구성들 거의 전부에서 KV cache 메모리 bandwidth-bound(그리고 HBM-bound)에 강하게 걸려 있다. generation throughput에 KV cache 크기 줄이기가 얼마나 중요한지를 잘 보여 주는 대목이다. latency/throughput 트레이드오프가 여기서도 얼마나 극적인지도 눈여겨보라.

</div>

<details>
<summary>코드 보기 — 꽤 간단하다</summary>

이 roofline들을 계산하는 코드는 다음과 같다:

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
bytes_per_param = 1  # int8 means 1 byte per param
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(
    num_chips: int,
    sequence_length: int,
    param_size: float,
) -> int:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  required_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(required_chips <= num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(
    num_chips=num_chips,
    sequence_length=sequence_length,
    param_size=param_size,
)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_size * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

latency를 KV 로딩과 파라미터 로딩이라는 두 원천으로 아주 명시적으로 분해했다는 점, 그리고 FLOPs와 comms 중 더 큰 쪽이 latency를 결정한다는 점을 눈여겨보라.

</details>

## 연습 문제

몇 가지 연습 문제다. 위에서 이미 다룬 내용과 겹치는 것도 있지만 학습에는 유용할 것이다.

**문제 1:** LLaMA 3-405B의 forward pass 한 번은 토큰당 몇 FLOPs를 쓰는가? FLOPs-bound라고 가정하면, TPU v5e의 N개 칩에서 forward pass 한 번에 걸리는 시간의 하한은 얼마인가? comms-bound라면? *모델이 칩 하나에 들어가지 않는다는 사실은 무시하라.*

**문제 2:** int8 weight와 int8 KV cache를 써서 LLaMA 3-8B를 BS240으로 서빙하고 싶다고 하자. (a) 모델 파라미터 (b) KV cache (c) peak 작업 activation은 (대략) 각각 몇 바이트를 쓰는가? 이것을 돌릴 수 있는 가장 작은 topology는 무엇인가?

**문제 3:** LLaMA 3-405B를 TPU v5e에서 어떻게 서빙하겠는가? int8 weight와 bfloat16 FLOPs를 가정한다. 토큰당 15ms라는 확고한 제한이 있다면, 달성할 수 있는 가장 높은 throughput 구성은 무엇인가? 이론적인 최소 step time은 얼마인가?

<div class="takeaway">

**8부는 여기까지!** XLA와 TPU 프로파일링을 깊이 파고드는 9부는 [여기](/scaling-book/profiling/)에서 볼 수 있다.

</div>

[^1]: 항상 참인 것은 아니다. 때로는 FLOPs보다 HBM이나 ICI bandwidth가 더 결정적이기도 하다. 그래도 좋은 휴리스틱이다.
