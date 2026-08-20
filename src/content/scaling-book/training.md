---
part: 5
title: "학습을 위한 Transformer 병렬화"
title_en: "How to Parallelize a Transformer for Training"
original: "https://jax-ml.github.io/scaling-book/training/"
summary: "LLM 학습에 쓰이는 네 가지 주요 병렬화 전략 — data parallelism, FSDP, tensor parallelism, pipeline parallelism — 을 다룬다. 각 전략의 통신 비용을 계산하고, 어느 지점부터 통신이 연산을 병목하기 시작하는지 유도한다."
date: 2026-08-20
published: true
---

> 여기서는 LLM 학습에 쓰이는 네 가지 주요 병렬화 전략을 다룬다: data parallelism, fully-sharded data parallelism(FSDP), tensor parallelism, 그리고 pipeline parallelism. 각 전략에 대해 어느 지점에서 통신에 의해 병목이 걸리는지 계산한다.

## 스케일링이란 무엇을 뜻하는가?

"모델 스케일링"의 목표는 학습이나 추론에 쓰는 칩 수를 늘렸을 때 throughput이 그에 비례해 선형으로 증가하게 만드는 것이다(이를 *strong scaling*이라고 부른다). 단일 칩에서의 성능은 메모리 bandwidth와 FLOPs 사이의 트레이드오프에 달려 있지만, 클러스터 수준의 성능은 칩 간 통신을 유용한 FLOPs와 겹쳐서 숨기는 데 달려 있다. 이는 쉽지 않은 일인데, 칩 수를 늘리면 통신 부담은 커지는 반면 통신을 숨기는 데 쓸 수 있는 디바이스당 연산량은 줄어들기 때문이다. [3장](/scaling-book/sharding/)에서 보았듯이 sharding된 행렬 곱셈은 비싼 AllGather나 ReduceScatter를 자주 요구하고, 이것이 TPU가 유용한 일을 하지 못하게 막을 수 있다. 이 장의 목표는 이 통신이 언제 *너무 비싸지는지* 알아내는 것이다.

이 장에서는 네 가지 흔한 병렬화 전략을 다룬다: (순수) **data parallelism**, **fully-sharded data parallelism**(FSDP / ZeRO sharding), **tensor parallelism**(model parallelism이라고도 부른다), 그리고 (간략하게) **pipeline parallelism**이다. 각 전략에 대해 어떤 통신 비용이 발생하는지, 그리고 그 비용이 어느 시점부터 연산 비용을 병목하기 시작하는지 보인다.[^1] 이 장에서는 칩 간 통신 비용에만 집중하면 되는데, 단일 칩의 batch size가 충분히 크기만 하면 HBM에서 MXU로의 데이터 전송은 이미 연산과 겹쳐지기 때문이다.

이 장 전체에서 계산을 단순하게 만들기 위해 다음 표기를 쓴다.

| 표기 | 의미 (모델 파라미터) |
| :------- | :--------------------------------------------------------------------- |
| D        | **d**<sub>model</sub> (hidden 차원/residual stream 차원)      |
| F        | **d**<sub>ff</sub> (feed-forward 차원)                        |
| B        | Batch 차원 (배치의 토큰 수; 디바이스당이 아니라 전체) |
| T        | 시퀀스 길이                                                        |
| L        | 모델의 레이어 수                                                  |

| 표기 | 의미 (하드웨어 특성) |
| :------- | :------------------------------------------------------------------------------------------------ |
| C        | 칩당 FLOPS/s                                                                                  |
| W        | 네트워크 bandwidth (양방향, 보통 $W_{\text{ici}}$나 $W_{\text{dcn}}$처럼 아래 첨자를 붙인다) |
| X        | mesh axis X 방향의 칩 수                                                                 |
| Y        | 또 다른 mesh axis Y 방향의 칩 수                                                           |
| Z        | 세 번째 mesh axis Z 방향의 칩 수                                                          |

단순함을 위해 **Transformer를 MLP 블록의 스택으로 근사한다** — [4장](/scaling-book/transformers/)에서 보았듯이 큰 모델에서 attention은 FLOPs의 비교적 작은 부분을 차지한다. gating matmul도 무시하면, 각 레이어는 다음의 단순한 구조로 남는다:

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/transformer-layer.png" alt="단순화된 Transformer 레이어" loading="lazy" />
  <figcaption><b>그림:</b> 단순화된 Transformer 레이어. 각 FFW 블록을 두 행렬 <b>W<sub>in</sub></b>: <code>bf16[D, F]</code>(up-projection)와 <b>W<sub>out</sub></b>: <code>bf16[F, D]</code>(down-projection)의 스택으로 취급하며, 입력은 <b>In</b>: <code>bf16[B, D]</code>이다.</figcaption>
</figure>

<details>
<summary>병렬화가 전혀 없는 우리의 작은 Transformer의 전체 알고리즘 보기</summary>

<div class="algorithm">

**Forward pass:** Loss[B]를 계산해야 한다

1.  Tmp[B, F] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

**Backward pass:** dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]를 계산해야 한다

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] \*<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] \*<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*이전 레이어들에 필요*)

</div>

통신이 추가된 알고리즘들과 비교할 수 있도록 이것을 제공해 둔다.

</details>

다음은 우리가 다룰 4가지 병렬화 전략이다. 각 전략은 위 다이어그램의 **In**, **W<sub>in</sub>, W<sub>out</sub>, Out**에 대한 sharding으로 유일하게 정의된다고 생각할 수 있다.

**1. Data parallelism:** *activation을 batch를 따라 shard하고, 파라미터와 optimizer state는 각 디바이스에 복제한다. 통신은 backward pass에서만 일어난다.*

$$
\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]
$$

**2. Fully-sharded data parallelism (FSDP 또는 ZeRO-3):** *activation은 (순수 data parallelism처럼) batch를 따라 shard하고, 파라미터는 같은 mesh axis를 따라 shard해 두었다가 forward pass에서 사용 직전에 just-in-time으로 AllGather한다. Optimizer state도 batch를 따라 shard한다. 중복 메모리를 줄인다.*

$$
\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]
$$

**3. Tensor parallelism (Megatron sharding 또는 model parallelism이라고도 부른다):** *activation을 D($d_\text{model}$)를 따라 shard하고, 파라미터를 F($d_{ff}$)를 따라 shard한다. 각 블록 전후로 activation을 AllGather·ReduceScatter한다. FSDP와 호환된다.*

$$
\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]
$$

**4. Pipeline parallelism:** *weight를 레이어 차원을 따라 shard하고, activation을 microbatch로 나눠 레이어 차원을 따라 흘려보낸다. pipeline stage 사이의 통신은 최소한이다(activation을 한 hop 옮기는 것뿐). 표기를 남용하면:*

$$
\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]
$$

### Data Parallelism

**문법:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

모델이 아주 작은 batch size(compute-bound가 되도록 240 토큰 이상)로도 단일 칩에 들어간다면, **항상 단순한 data parallelism을 써야 한다.** 순수 data parallelism은 TPU 수가 batch size보다 작기만 하면 activation을 임의의 수의 TPU에 나눠 담는다. forward pass에는 통신이 전혀 없지만, 매 스텝의 끝에 **각 TPU는 파라미터를 업데이트하기 전에 로컬 gradient를 동기화하기 위해 AllReduce를 수행한다.**

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/data-parallelism.png" alt="순수 data parallelism 다이어그램" loading="lazy" />
  <figcaption><b>그림:</b> 순수 data parallelism의 다이어그램(forward pass). activation(왼쪽)은 batch 차원을 따라 완전히 shard되고 weight는 완전히 복제되어, 각 TPU가 동일한 weight 사본을 가진다. 이는 weight의 총 메모리가 N배로 늘어난다는 뜻이지만, forward pass에는 통신이 전혀 필요 없다.</figcaption>
</figure>

<details>
<summary>forward·backward pass의 전체 알고리즘 보기 (순전히 간결함을 위해 dL/dOut을 dOut으로 줄여 쓰는 표기 남용을 허용한다)</summary>

<div class="algorithm">

**순수 Data Parallelism 알고리즘:**

**Forward pass:** Loss[B<sub>X</sub>]를 계산해야 한다

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**Backward pass:** dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]를 계산해야 한다

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*critical path에 있지 않아 비동기로 수행 가능*)
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*critical path에 있지 않아 비동기로 수행 가능*)
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*이전 레이어들에 필요*)

</div>

loss 함수의 세부는 무시하고 $\text{Tmp} = W_\text{in} \cdot \text{In}$으로 줄여 쓴다. 최종 loss는 평균 **AllReduce**(Loss[B<sub>X</sub>])이지만, weight gradient를 평균할 때 backward pass에서만 AllReduce를 계산하면 된다는 점에 유의하라.

</details>

forward pass에는 통신이 없다는 점에 주목하라 — **전부 backward pass에 있다**! backward pass에는 또 하나의 훌륭한 성질이 있는데, AllReduce들이 "critical path"에 있지 않다는 것이다. 즉 각 AllReduce는 편한 시점에 아무 때나 수행할 수 있고, 이후 연산의 수행을 막지 않는다. 전체 통신 비용이 총 연산 비용을 넘어서면 _여전히 병목이 될 수는 있지만_, 구현 관점에서는 훨씬 너그럽다. model/tensor parallelism에는 이 성질이 없다는 것을 곧 보게 된다.

**왜 쓰는가?** 순수 data parallelism은 activation을 batch 차원으로 쪼개 activation 메모리 부담을 줄이므로, batch 차원을 나눠 담을 칩만 더 있으면 batch size를 거의 마음대로 키울 수 있다. 특히 학습 중에는 activation이 메모리 사용량을 지배하는 경우가 많아 매우 유용하다.

**왜 쓰지 않는가?** 순수 data parallelism은 모델 파라미터나 optimizer state의 메모리 부담은 전혀 줄여 주지 않는다. 즉 파라미터 + optimizer state가 단일 TPU에 들어가지 않는, 규모 있는 흥미로운 모델에는 순수 data parallelism이 유용한 경우가 드물다. 규모 감각을 위해: bf16 파라미터와 Adam의 fp32 optimizer state로 학습하면[^2] 담을 수 있는 가장 큰 모델은 $$\text{TPU memory} / 10$$ 파라미터이므로, 예컨대 96GB HBM의 TPUv5p 칩에서 순수 data parallelism이면 약 9B 파라미터다.

<div class="takeaway">

**요점(Takeaway):** Adam과 순수 data parallelism으로 학습할 수 있는 가장 큰 모델은 $$\text{num\_params} = \text{HBM per device} / 10$$이다. TPU v5p에서는 대략 9B 파라미터다.[^3]

</div>

*학습에서 실제 모델에 유용해지려면, 모델 파라미터나 optimizer를 적어도 부분적으로는 shard해야 한다.*

**언제 통신에 의해 병목이 걸리는가?** 위에서 보았듯이 레이어당 AllReduce가 두 번 있고, 각각의 크기는 (bf16 weight 기준) $$2DF$$다. data parallelism은 언제 우리를 communication-bound로 만드는가?

위 표에서처럼 $C$ = 칩당 FLOPs, $W_{\text{ici}}$ = **양방향** 네트워크 bandwidth, $X$ = batch를 분할하는 shard 수라고 하자[^4]. 관련 matmul을 수행하는 데 필요한 시간 $$T_\text{math}$$와 필요한 통신 시간 $$T_\text{comms}$$를 계산해 보자. 이 병렬화 전략은 forward pass에 통신이 없으므로 backward pass에 대해서만 이 값들을 계산하면 된다.

*통신 시간:* 앞 장에서 보았듯이 1D mesh에서 AllReduce를 수행하는 데 필요한 시간은 AllReduce되는 배열의 총 바이트와 ICI bandwidth $W_\text{ici}$에만 의존한다. 구체적으로 AllReduce 시간은 $2 \cdot \text{total bytes} / W_\text{ici}$다. $W_\text{in}$과 $W_\text{out}$ 둘 다 AllReduce해야 하므로 레이어당 AllReduce는 2번이다. 각 AllReduce는 weight 행렬 하나, 즉 $DF$개 파라미터 = $2DF$ 바이트짜리 배열에 대한 것이다. 종합하면 한 레이어의 AllReduce 총 시간은

$$
\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}
$$

*Matmul 시간:* 각 레이어는 forward pass에서 matmul 2번, backward pass에서 matmul 4번으로 구성되고, 각각 $2(B/X)DF$ FLOPs를 요구한다. 따라서 backward pass의 한 레이어에 대해

$$
\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}
$$

통신과 연산을 겹치므로 레이어당 총 시간은 두 값의 max다:

$$
\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}
$$

$T_\text{math}/T_\text{comms} > 1$일 때, 즉 다음일 때 compute-bound가 된다:

$$
\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}
$$

요컨대 data parallelism으로 compute-bound를 유지하려면 디바이스당 batch size $$B / X$$가 ICI operational intensity $C / W_\text{ici}$를 넘어야 한다. 이는 결국 연산 시간은 디바이스당 batch size에 비례해 커지는 반면 (모델 weight를 전송하므로) 통신 시간은 이 값과 무관하다는 사실의 귀결이다. $B/X > C/W_\text{ici}$ 조건이 단일 디바이스의 compute-bound 규칙 $B > 240$과 닮았다는 점에 주목하라. 그 경우에도 규칙은 연산 시간이 batch size에 비례하는 반면 데이터 전송량은 ($B \ll F, D$ 영역에서) batch size와 무관하다는 사실에서 나왔다.

감을 잡기 위해 실제 수치를 넣어 보자. TPUv5p에서 ICI를 통한 1D data parallelism이면 `C=4.6e14`, `W=2 * 9e10`이므로 **communication-bound를 피하려면 칩당 batch size가 최소 2,550이어야 한다**. data parallelism은 여러 axis에 걸쳐 할 수 있으므로, TPUv5p pod의 세 axis 전부를 순수 data parallelism에 바치면 bandwidth $W_\text{ici}$가 3배가 되어 TPU당 BS=850, 즉 (8960칩) pod당 배치 7.6M 토큰까지 내려갈 수 있다! **이는 순수 data parallelism으로는 병목에 걸리기가 꽤 어렵다는 것을 말해 준다!**

<div class="takeaway">

**참고 [context parallelism]:** 이 장 전체에서 $B$는 항상 **토큰 단위의** 전체 batch size를 가리킨다. 그런데 분명 우리 배치는 여러 다른 시퀀스로 이루어져 있는데, 이게 어떻게 성립할까? MLP의 입장에서는 **토큰은 그냥 토큰이다**! 같은 시퀀스에 속하든 서로 다른 두 시퀀스에 속하든 상관없다. 따라서 batch 차원과 시퀀스 차원 둘 다에 대해 어느 정도 자유롭게 data parallelism을 할 수 있다: 이를 context parallelism 또는 sequence parallelism이라고 부르지만, 그냥 또 다른 종류의 data parallelism이라고 생각해도 된다. attention은 시퀀스를 가로지르는 연산이 있어 MLP보다 까다롭지만, attention 중에 KV나 Q를 gather하고 FLOPs와 통신을 신중하게 겹치는 것으로 처리할 수 있다(보통 "ring attention"이라는 것을 쓴다). 이 장에서는 시퀀스 차원을 아예 무시하고 batch 또는 sequence parallelism이 어느 정도 있다고 가정한다.

</div>

**여러 mesh axis에 대한 참고:** 주어진 병렬화 전략에 여러 mesh axis를 쓰면 가용 bandwidth가 어떻게 달라지는지 짚고 넘어가자. 여러 mesh axis를 쓰면 더 많은 bandwidth를 얻는다.

* **정의:** $M_X$($M_Y$, $M_Z$ 등)는 주어진 병렬화 전략이 걸쳐 있는 하드웨어 mesh axis의 수다.
* **효과 (bandwidth-bound일 때):** $M$개의 axis를 쓰면 (약 $M$배의) 합산 링크 bandwidth를 얻으므로, collective 시간은 $\propto 1/M_X$로 줄어든다.

### Fully-Sharded Data Parallelism (FSDP)

**문법:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

Fully-sharded data parallelism(흔히 FSDP 또는 ZeRO-sharding이라고 부른다 (Rajbhandari et al. 2019))은 모델의 optimizer state와 weight를 data parallel shard들에 나눠 담고, 필요할 때 효율적으로 gather·scatter한다. **순수 data parallelism에 비해 FSDP는 디바이스당 메모리 사용량을 극적으로 줄이고 backward pass FLOPs도 아끼며, 오버헤드는 아주 작다.**

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/fsdp.png" alt="FSDP 다이어그램" loading="lazy" />
  <figcaption><b>그림:</b> FSDP는 W<sub>in</sub>의 contracting 차원과 W<sub>out</sub>의 출력 차원을 data 차원을 따라 shard한다. 이는 메모리를 줄이지만 (3장에서 보았듯) matmul을 수행하기 전에 W의 weight를 gather하도록 강제한다. activation(왼쪽)은 <i>contracting 차원을 따라 shard되어 있지 않으며</i>, 바로 이것이 우리가 gather를 해야 하는 이유다. <b>weight의 optimizer state도 마찬가지로 contracting 차원을 따라 shard된다는 점에 유의하라.</b></figcaption>
</figure>

([3장](/scaling-book/sharding/)에서) AllReduce가 AllGather와 ReduceScatter로 분해될 수 있다는 것을 기억할 것이다. 이는 표준 data parallelism처럼 전체 gradient AllReduce를 하는 대신, weight와 optimizer state를 칩들에 shard해 두고 forward pass에서 레이어마다 AllGather하고 backward pass에서 weight에 대해 ReduceScatter해도 추가 비용이 전혀 없다는 뜻이다.

<details>
<summary>FSDP의 전체 알고리즘 보기</summary>

<div class="algorithm">

**Fully-Sharded Data Parallelism (FSDP):**

**Forward pass:** Loss[B<sub>X</sub>]를 계산해야 한다

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*critical path에 있지 않아 이전 레이어를 계산하는 동안 수행 가능*)
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] (*이제 W<sub>in</sub>[D, F]를 버려도 된다*)
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*critical path에 있지 않아 이전 레이어를 계산하는 동안 수행 가능*)
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**Backward pass:** dW<sub>out</sub>[F, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F]를 계산해야 한다

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*critical path에 있지 않아 비동기로 수행 가능*)
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*미리 수행해 둘 수 있다*)
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] *(여기서 W<sub>out</sub>[F, D]를 버려도 된다)*
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) *(critical path에 있지 않아 비동기로 수행 가능)*
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*미리 수행해 둘 수 있다*)
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*이전 레이어들에 필요) (여기서 W<sub>in</sub>[D, F]를 버려도 된다*)

</div>

</details>

이는 "ZeRO Sharding"이라고도 불리는데, 불필요한 연산이나 불필요한 state 저장을 전혀 하지 않는다는 뜻의 "Zero Redundancy Optimizer"에서 온 이름이다. ZeRO-{1,2,3}은 각각 optimizer state, gradient, weight를 이런 식으로 shard하는 것을 가리킨다. 셋 다 통신 비용이 같으므로[^5] 기본적으로 항상 ZeRO-3 sharding을 하면 되고, 이는 파라미터·gradient·optimizer state를 디바이스 집합에 걸쳐 shard한다.

**왜 이렇게 하는가?** 표준 data parallelism에는 중복 작업이 많다. 각 TPU가 전체 gradient를 AllReduce한 다음, 전체 optimizer state를 업데이트하고(모든 TPU에서 동일한 작업), 파라미터를 업데이트한다(역시 완전히 중복). ZeRO sharding(gradient/optimizer state를 shard)이라면, AllReduce 대신 gradient를 ReduceScatter하고, 자신의 optimizer state shard만 업데이트하고, 파라미터의 shard 하나를 업데이트한 뒤, forward pass에 필요해질 때 파라미터를 AllGather하면 된다.

**언제 통신에 의해 병목이 걸리는가?** FLOPs 대 통신의 상대 비용은 순수 data parallelism과 정확히 같다. backward pass의 각 AllReduce가 AllGather + ReduceScatter로 바뀌었을 뿐이기 때문이다. AllReduce는 각각 비용이 절반인 AllGather와 ReduceScatter로 구현된다는 것을 기억하라. 여기서는 forward pass를 모델링하는데, backward pass와 FLOPs 대 통신 비율이 같기 때문이다:

$$
\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}
$$

따라서 순수 data parallelism과 마찬가지로 $$B / X > C / W_\text{ici}$$일 때, 즉 디바이스당 batch size $B/X$가 "ICI operational intensity" $C/W_\text{ici}$(v5p에서 `4.59e14 / 1.8e11 = 2550`)를 넘을 때 compute-bound다. 이는 아주 좋은 소식인데, 순수 data parallelism에서 compute-bound가 될 만큼 디바이스당 batch size가 크다면 — compute-bound 영역을 벗어날 걱정 없이 — 그냥 FSDP로 업그레이드해서 파라미터와 optimizer state 메모리를 어마어마하게 아낄 수 있다는 뜻이기 때문이다! forward pass에 통신을 더하긴 했지만, 이 비용은 forward pass의 FLOPs와 그냥 겹쳐지므로 문제가 되지 않는다.

<div class="takeaway">

**요점(Takeaway):** FSDP와 순수 Data Parallelism 모두 TPUv5에서 디바이스당 batch size가 $2550 / M_X$보다 작아지면 bandwidth bound가 된다. 여기서 $M_X$는 mesh axis 수다.

</div>

예를 들어 DeepSeek-V2(학습 batch size 정보를 공개한 몇 안 되는 최근의 강력한 모델 중 하나)는 약 40M 토큰의 batch size를 썼다. **이 정도면 bandwidth 한계에 부딪히기 전까지 대략 47,000개의 칩, 약 5개의 TPUv5 pod까지 스케일할 수 있다.**

약 `6.3e24 (15e12 * 70e9 * 6)` FLOPs로 학습된 LLaMA-3 70B라면, 16M 토큰짜리 배치를 대략 `16e6 / (2550 / 3) = 18,823`개의 칩(8960칩 pod 약 2개)에 나눌 수 있고, 각 칩이 `4.59e14` FLOPs를 peak FLOPs 활용률(흔히 MFU라고 부른다) 50%로 돌리면 **약 17일 만에 학습할 수 있다**. 나쁘지 않다! 그렇지만 어떻게 더 잘할 수 있는지 알아보자.

<div class="takeaway">

**Critical batch size에 대한 참고:** 다소 직관에 반하게, (칩 수를 고정하면) 전체 batch size가 줄어들수록 통신 병목은 더 심해진다. Data parallelism과 FSDP는 batch size를 계속 키울 수만 있다면 임의의 수의 칩까지 스케일하게 해 준다! 하지만 실전에서는 batch size가 커질수록 gradient가 거의 noise-free해져서 학습의 수익이 체감하는 경향이 있다. 학습 불안정이 나타나기도 한다. 따라서 "무제한 컴퓨트 영역"에서 최적의 sharding 방식을 찾는 게임은 보통, scaling law로 정해진 고정 batch size와 이미 알고 있는 (큰) 칩 수에서 출발해서, 그 작은 batch size를 그 많은 칩에 얹을 수 있게 해 주는 분할을 찾는 것을 목표로 한다.

</div>

### Tensor Parallelism

**문법:** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$ (나중에 FSDP와 조합하기 위해 $$Y$$를 쓴다)

fully-sharded data parallelism의 AllReduce에서는 weight를 칩 사이에서 옮긴다. 대신 모델의 feedforward 차원을 shard하고 레이어 도중에 activation을 옮길 수도 있다 — 이를 "1D model parallelism" 또는 Megatron sharding (Shoeybi et al. 2019)이라고 부른다. 이는 pod당 효율적인 batch size를 더 작게 만들어 줄 수 있다. 아래 그림은 행렬 하나를 이런 식으로 shard한 예다:

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/model-parallelism.png" alt="기본적인 tensor parallelism의 예" loading="lazy" />
  <figcaption><b>그림:</b> 기본적인 tensor parallelism의 예. activation을 Y에 대해서만 shard하므로(X에 대해 shard하는 FSDP와 달리) activation은 X에 대해 복제된다. 표준 문법으로 쓰면 <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]이다. contracting 차원 중 하나만 shard되어 있으므로, 보통 matmul 전에 activation <b>A</b>를 AllGather한다.</figcaption>
</figure>

앞서 언급했듯, **In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\]는 첫 matmul 전에 activation을 gather해야 한다는 뜻이다. 이는 activation이 weight보다 작을 때 ZeRO sharding보다 싸다.** 이는 보통 어느 정도의 ZeRO sharding이 더해져 있을 때만 성립한다(gather의 크기를 줄여 주기 때문이다). 우리가 ZeRO sharding과 tensor parallelism을 섞어 쓰는 경향이 있는 이유 중 하나다.

<details>
<summary>tensor parallelism의 알고리즘 보기</summary>

<div class="algorithm">

**Tensor Parallelism:**

**Forward pass:** Loss[B]를 계산해야 한다

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(critical path에 있음)*
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(contracting 차원으로 shard되어 있지 않으므로 통신 없음)*
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>}) *(critical path에 있음)*
5.  Loss[B] = ...

**Backward pass:** dW<sub>out</sub>[F<sub>Y</sub>, D], dW<sub>in</sub>[D, F<sub>Y</sub>]를 계산해야 한다

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) *(critical path에 있음)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(여기서 dOut[B, D]를 버려도 된다)*
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(forward pass의 (1)과 공유하면 건너뛸 수 있다)*
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = In[B, D] \*<sub>B</sub> dTmp[B, F<sub>Y</sub>]
7.  dIn[B, D] {U<sub>Y</sub>} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(이전 레이어들에 필요)*
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U<sub>Y</sub>}) *(critical path에 있음)*

</div>

</details>

tensor parallelism의 좋은 점 하나는 Transformer forward pass의 두 행렬과 잘 맞물린다는 것이다. 순진하게 하면 두 행렬 각각 뒤에 AllReduce를 하게 될 것이다. 하지만 여기서는 먼저 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**를 하고, 그다음 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**를 한다. 즉 AllReduce를 하는 대신, 처음에 **In**을 AllGather하고 끝에 **Out**을 ReduceScatter하면 된다.

**비용은 얼마나 드는가?** forward pass만 모델링하자 — backward pass는 여기 나오는 각 연산의 transpose일 뿐이다. 1D tensor parallelism에서는 첫 matmul 전에 activation을 AllGather하고 두 번째 matmul 후에 ReduceScatter하며, 한 번에 2바이트(bf16)씩 보낸다. 언제 통신에 병목이 걸리는지 알아보자.

$$
\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}
$$

연산 비용이 통신 비용보다 커야 한다는 점에 주목하면:

$$
\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}
$$

$$
\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}
$$

$$
\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}
$$

따라서 예컨대 TPUv5p에서는 bf16 기준 $C / W_{ici} = 2550$이므로 tensor parallelism은 $Y < F / 2550$까지만 할 수 있다. ICI axis가 여러 개라면 $T_\text{comms}$가 $M_Y$배 줄어들어 $Y < M_Y \cdot F / 2550$이 된다.

<div class="takeaway">

**요점(Takeaway):** Tensor Parallelism은 $Y > M_Y \cdot F / 2550$이면 communication bound가 된다. 대부분의 모델에서 이는 8-way에서 16-way tensor parallelism 사이다.

</div>

**이 조건이 연산 정밀도에 의존하지 않는다는 점에 유의하라.** 예컨대 TPUv5p에서 int8이면 $$C_\text{int8} / W_{ici}$$가 $$2550$$ 대신 $$5100$$이지만 통신량도 절반이 되어, 두 개의 2배 요인이 상쇄된다.

**몇 가지 예를 생각해 보자:**

* TPUv5p에서 $$D = 8192,$$ $$F \approx 30,000$$인 LLaMA 3-70B는 8-way tensor parallelism은 무리 없이 할 수 있지만, 16-way tensor parallelism에서는 communication bound가 된다. 8-way model sharding에 필요한 F는 20k다.

* Gemma 7B는 $$F \approx 50k$$이므로 19-way tensor parallelism에서야 communication bound가 된다. 즉 16-way까지는 여전히 좋은 성능을 볼 수 있을 것이다.

### FSDP와 Tensor Parallelism 조합하기

**문법:** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP와 tensor parallelism의 좋은 점은 둘을 조합할 수 있다는 것이다. **W<sub>in</sub>**과 **W<sub>out</sub>**을 두 axis 모두를 따라 shard하면 메모리와 연산을 둘 다 아낀다. B를 X를 따라 shard하므로 model-parallel AllGather의 크기가 줄고, F를 Y를 따라 shard하므로 FSDP의 통신 오버헤드가 준다. 즉 둘을 조합하면 위에서 본 것보다 더 낮은 유효 batch size까지 갈 수 있다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/mixed-fsdp-model-parallelism.png" alt="FSDP와 tensor parallelism을 조합한 다이어그램" loading="lazy" />
  <figcaption><b>그림:</b> FSDP와 tensor parallelism을 조합한 다이어그램. 다른 경우들과 달리 모델 파라미터의 중복이 전혀 없다.</figcaption>
</figure>

<details>
<summary>혼합 FSDP + tensor parallelism의 전체 알고리즘 보기 (통신은 많지만, activation을 batch로 shard하고 weight를 tensor로 훨씬 많이 shard한 덕에 AllGather와 ReduceScatter가 전부 더 작다!)</summary>

<div class="algorithm">

**Forward pass:** Loss[B]를 계산해야 한다

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(critical path에 있음)*
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(미리 수행해 둘 수 있다)*
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(미리 수행해 둘 수 있다)*
5.  Out[B<sub>X</sub>, D] {U<sub>Y</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U<sub>Y</sub>}) *(critical path에 있음)*
7.  Loss[B<sub>X</sub>] = ...

**Backward pass:** dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]를 계산해야 한다

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) *(critical path에 있음)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U<sub>X</sub>})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(미리 수행해 둘 수 있다)*
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(여기서 dOut[B, D]를 버려도 된다)*
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(critical path에 있지 않음 + 이전 레이어의 (2)와 공유할 수 있다)*
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F<sub>Y</sub>]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U<sub>X</sub>})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(미리 수행해 둘 수 있다)*
11. dIn[B<sub>X</sub>, D] {U<sub>Y</sub>} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(이전 레이어들에 필요)*
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U<sub>Y</sub>}) *(critical path에 있음)*

</div>

</details>

**FSDP와 TP의 올바른 조합은 무엇인가?** 단순하지만 핵심적인 격언은, FSDP는 weight를 옮기고 tensor parallelism은 activation을 옮긴다는 것이다. 즉 batch size가 줄어들수록(특히 data parallelism을 더 할수록) shard당 activation이 작아져 tensor parallelism이 싸진다.

* Tensor parallelism은 $$\mathbf{AllGather}_Y([B_X, D_Y])$$를 수행하는데, 이는 $$X$$가 커질수록 작아진다.
* FSDP는 $$\mathbf{AllGather}_X([D_X, F_Y])$$를 수행하는데, 이는 $$Y$$가 커질수록 작아진다.

따라서 둘을 조합하면 replica당 최소 batch size를 더욱더 낮출 수 있다. 위와 같은 방식으로 FSDP와 TP의 최적량을 계산할 수 있다:

$X$를 FSDP에 배정한 칩 수, $$Y$$를 tensor parallelism에 배정한 칩 수라고 하자. $$N$$은 슬라이스의 총 칩 수이고 $$N=XY$$다. $$M_X$$와 $$M_Y$$는 FSDP와 TP를 각각 수행하는 mesh axis 수다(대략 합해서 3이 되어야 한다). FLOP당 통신이 가장 많은 forward pass만 모델링한다. 위 알고리즘의 통신을 합산하면

$$
T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}
$$

$$
T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}
$$

마찬가지로 총 FLOPs 시간은

$$
T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.
$$

분석을 단순화하기 위해 두 가지를 가정한다: 첫째, $X$와 $Y$가 (양수이고 $XY=N$을 만족하는 한) 정수가 아닌 값도 가질 수 있게 허용한다. 둘째, $X$ axis와 $Y$ axis의 통신을 서로 완전히 겹칠 수 있다고 가정한다. 두 번째 가정 아래 총 통신 시간은

$$
T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)
$$

어떤 조건에서 compute-bound가 되는지 묻기 전에, 총 통신을 최소화하는 $X$와 $Y$의 최적값부터 찾자. FLOPs는 $X$와 $Y$에 무관하므로, 최적 설정은 그저 통신을 최소화하는 설정이다. 이를 위해 위의 $T_\text{comms}$를 $X$와 $Y$ 대신 $X$와 (시스템의 칩 수라서 고정된) $N$으로 다시 쓰자:

$$
T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)
$$

$T_\text{FSDP comms}$는 $X$에 대해 단조 증가하고 $T_\text{TP comms}$는 $X$에 대해 단조 감소하므로, max는 $T_\text{FSDP comms} = T_\text{TP comms}$일 때 최소가 되고, 이는 다음일 때 성립한다

$$
\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}
$$

이건 정말 유용하다! 주어진 $B$, $F$, $N$에 대해 어느 정도의 FSDP가 최적인지 알려 준다. 감을 잡아 보자. 현실적인 값, 즉 $N = 64$(4x4x4 칩 배열에 해당), $B=48,000$, $F=32768$을 넣으면 대략 $X\approx 13.9$가 나온다. 그러면 계산된 최적값에 가깝게 $X$는 16, $Y$는 4로 잡을 것이다.

<div class="takeaway">

**요점(Takeaway):** 일반적으로 학습 중 최적의 FSDP 양은 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$이다.

</div>

이제 모든 병렬화 전략에 물어 온 질문으로 돌아가자: **어떤 조건에서 compute-bound가 되는가?** FLOPs와 통신을 겹칠 수 있으므로, 다음일 때 compute-bound다

$$
\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}
$$

$\alpha \equiv C / W_\text{ici}$, 즉 ICI arithmetic intensity로 두면 다음처럼 단순화할 수 있다:

$$
\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}
$$

$X_{opt}$는 좌변 max의 두 항이 같아지도록 계산한 값이므로, 어느 한쪽에 그냥 대입하면 된다($Y_{opt} = N/X_{opt}$에 유의). 즉

$$
\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}
$$

더 단순화하면

$$
\sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},
$$

여기서 좌변은 통신 시간에 비례하고 우변은 연산 시간에 비례한다. 연산 시간은 (병렬화와 무관하게 늘 그렇듯) batch size에 선형으로 비례하는 반면, 통신 시간은 batch size의 제곱근에 비례한다는 점에 주목하라. 따라서 연산 대 통신 시간의 비율도 batch size의 제곱근으로 커진다:

$$
\frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}.
$$

이 비율이 1보다 커서 compute bound가 되려면 다음이 필요하다

$$
\frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}
$$

대략적인 수치를 얻기 위해 다시 $F=32,768$, $\alpha=2550$, $M_X M_Y=2$(3D mesh라면 그럴 수밖에 없다)를 넣자. 그러면 대략 $B/N > 99$가 나온다. 순수 data parallel(또는 FSDP)의 경우 3D mesh 가정에서 $B/N$이 약 $850$을 넘어야 compute bound였던 것과 비교하면, 대략 8배를 벌어들인 셈이다.

<div class="takeaway">

**요점(Takeaway):** tensor parallelism과 FSDP를 조합하면 $B/N$을 $$2550^2 / 2F$$까지 낮출 수 있다. 이는 칩당 배치를 100 정도까지 감당할 수 있다는 뜻으로, FSDP만으로 가능한 것보다 대략 8배 작다.

</div>

아래는 대표적인 4x4x4 칩 배열에서 혼합 FSDP + TP의 FLOPs 대 통신 시간 비율을, tensor parallelism만(TP) 쓰는 경우 및 data parallelism만(FSDP) 쓰는 경우와 비교해 그린 것이다. 아주 큰 batch size에서는 순수 FSDP가 지배적이지만, 칩 수 대비 batch size가 대략 100에서 850 사이인 영역에서는 compute-bound가 되기 위해 혼합 FSDP + TP 전략이 필요하다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/mixed-fsdp-comms-2.png" alt="최적 혼합 FSDP/TP의 FLOPs 대 통신 비율" loading="lazy" />
</figure>

*<b>그림:</b> F=30k인 TPUv5p 4x4x4 슬라이스에서 최적 혼합 FSDP/TP의 FLOPs 대 통신 시간 비율. 예상대로 tensor parallelism은 batch size와 무관한 고정 비율을 가진다. 이상적인 혼합 FSDP + TP는 $\sqrt{B}$로, FSDP는 $B$로 스케일한다. 그러나 중간 batch size 영역에서는 FSDP + TP만이 1보다 큰 비율을 달성한다.*

다음은 TPU v5p 16x16x16에서 여러 sharding 방식에 대해 batch size에 따른 FLOPs 시간과 통신 시간을 보여 주는 또 다른 예다.

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/math-comms-time.png" alt="병렬화 방식별 통신 시간" loading="lazy" />
  <figcaption><b>그림:</b> 서로 다른 병렬화 방식에서 통신에 걸리는 시간. 검은 점선은 행렬 곱셈 FLOPs에 걸리는 시간이므로, 이 선보다 위에 있는 곡선은 모두 comms-bound다. 모든 전략이 batch size 6e5 아래에서 comms-bound가 되는데, 이는 예상값 4096 * 2550^2 / (2 * 8192 * 4) = 4e5와 부합한다.</figcaption>
</figure>

검은 곡선은 모델 FLOPs에 쓰는 시간이므로, 이것이 모든 통신 비용보다 낮은 batch size에서는 엄격히 comms bound다. 검은 곡선이 초록 곡선과 약 `4e5`에서 교차하는 것을 볼 수 있는데, 예측한 그대로다.

다음은 여러 batch size에 대한 총 연산 시간과 통신 시간을 직접 만져 볼 수 있는 인터랙티브 애니메이션이다:

<div class="l-page">
  <iframe src="https://jax-ml.github.io/scaling-book/assets/plotly/training-roofline.html" frameborder="0" scrolling="no" height="400px" width="100%"></iframe>
</div>

대체로 위 내용과 일치한다는 것을 볼 수 있다(최솟값이 FSDP=256, TP=16 부근). 각 전략의 axis 수가 조금씩 달라서 약간의 오차는 있다.

### Pipelining

앞 절들에서 pipelining 이야기를 아예 피해 왔다는 것을 눈치챘을 것이다. pipelining은 GPU 병렬화에서는 지배적인 전략이지만 TPU에서는 다소 덜 필수적이다. 간단히 말해, pipeline 학습은 모델의 레이어들을 여러 디바이스에 나누고 forward·backward pass 동안 pipeline stage 사이에서 activation을 전달하는 것이다. 알고리즘은 대략 다음과 같다:

1. 데이터를 TPU 0에 초기화하고, weight는 레이어 차원을 따라 shard한다(FSDP·tensor parallelism까지 함께 쓰는 pipelining이면 $W_\text{in}[L_Z, D_X, F_Y]$).
2. 첫 레이어를 TPU 0에서 수행한 뒤, 결과 activation을 TPU 1로 복사하고, 마지막 TPU에 도달할 때까지 반복한다.
3. loss 함수와 그 도함수 $\partial L / \partial x_L$을 계산한다.
4. 마지막 pipeline stage에서 도함수 $\partial L / \partial W_L$과 $\partial L / \partial x_{L-1}$을 계산한 뒤, $\partial L / \partial x_{L-1}$을 이전 pipeline stage로 복사하고, TPU 0에 도달할 때까지 반복한다.

<details>
<summary>(동작하는) Python 의사코드 보기</summary>

이 의사코드는 Cloud TPU VM에서 실행할 수 있다. 아주 효율적이거나 현실적이지는 않지만, 데이터가 디바이스들을 거쳐 어떻게 전파되는지 감을 준다.

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# Pretend each layer is just a single matmul.
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# Assume we have num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # make up some fake loss function

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(num_layers - 1, -1, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i], weights[i])
  dx, dw = f_vjp(dx)  # compute the jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # update our weights

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

</details>

**왜 좋은 아이디어인가?** pipelining이 훌륭한 이유는 많다: pipeline stage 사이의 통신 비용이 낮아서, bandwidth가 낮은 interconnect로도 아주 큰 모델을 학습할 수 있다. GPU는 TPU처럼 ICI로 조밀하게 연결되어 있지 않기 때문에, GPU에서 특히 유용한 경우가 많다.

**왜 어렵고 성가신가?** 위 의사코드에서 TPU 0이 거의 항상 놀고 있다는 것을 눈치챘을 것이다! TPU 0은 pipeline의 맨 첫 스텝과 맨 마지막 스텝에서만 일한다. 이 유휴 기간을 pipeline bubble이라고 부르며, 다루기가 아주 성가시다. 보통은 먼저 microbatching으로 이를 완화하는데, 여러 개의 작은 배치를 pipeline에 흘려보내 TPU 0이 적어도 전체 스텝 시간의 더 큰 비율 동안은 활용되게 한다.

두 번째 접근은 forward matmul $W_i @ x_i$, backward의 $dx$ matmul $W_i @ \partial L / \partial x_{i+1}$, 그리고 $dW$ matmul $\partial L / \partial x_{i+1} @ x_i$를 신중하게 겹치는 것이다. 각각이 어느 정도 FLOPs를 요구하므로, 이들을 겹쳐서 bubble을 완전히 숨길 수 있다. 다음은 최근 DeepSeek v3 논문 (DeepSeek-AI et al. 2024)에 실린 "bubble-free" pipeline 스케줄 그림이다:

<figure>
  <img src="https://jax-ml.github.io/scaling-book/assets/img/deepseek-pipeline.png" alt="DeepSeek v3 pipeline 스케줄" loading="lazy" />
  <figcaption><b>그림:</b> DeepSeek v3의 pipeline 스케줄 (<a href="https://arxiv.org/pdf/2412.19437">최근 논문</a>에서). 주황색은 forward matmul, 초록색은 dL/dx matmul, 파란색은 dL/dW matmul이다. backward의 dL/dx 곱셈을 우선시함으로써 FLOPs가 "발이 묶이는(stranding)" 것을 피할 수 있다.</figcaption>
</figure>

(더 크게 상호연결된 pod를 가진) TPU에는 덜 결정적이므로 여기서 더 깊이 파고들지는 않겠지만, 핵심적인 pipelining 병목을 이해해 보는 것은 좋은 연습이다.

### Pod를 넘어 스케일링하기

가능한 가장 큰 TPU 슬라이스는 8960칩(호스트 2240개)의 TPU v5p SuperPod다. 이 크기를 넘어 스케일하고 싶으면 Data-Center Networking(DCN) 경계를 건너야 한다. 각 TPU 호스트에는 호스트를 이더넷으로 다른 TPU v5p pod들에 연결하는 NIC(Network Interface Card)가 하나 이상 달려 있다. [TPU 장](/scaling-book/tpus/)에서 언급했듯 각 호스트는 약 200Gbps(25GB/s)의 full-duplex DCN bandwidth를 가지며, 이는 TPU당 약 6.25GB/s의 full-duplex(egress) bandwidth다.

보통 단일 pod를 넘어 스케일할 때는 ICI 도메인 안에서 어떤 형태의 model parallelism이나 FSDP를 하고, 여러 pod에 걸쳐서는 순수 data parallelism을 한다. $N$을 스케일하려는 TPU 수, $M$을 ICI로 연결된 슬라이스당 TPU 수라고 하자. DCN 위에서 AllReduce를 하려면 pod들의 집합에 대해 ring-reduction을 하면 되고, (backward pass에서) 다음을 얻는다:

$$
T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}
$$

$$
T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}
$$

통신 bandwidth는 $M$에 비례해 커지는데, ICI와 달리 ICI 도메인을 키우면 NIC를 더 많이 확보해 총 bandwidth가 커지기 때문이다. 단순화하면 $T_\text{math} > T_\text{comms}$는 다음일 때다

$$
\frac{B}{\text{slice}} > \frac{C}{W_\text{dcn}}
$$

TPU v5p에서 $\frac{C}{W_\text{dcn}}$은 약 `4.59e14 / 6.25e9 = 73,440`이다. 즉 DCN 위에서 효율적으로 스케일하려면, 각 노드에서 egress하기 위해 필요한 ICI 도메인당 최소 batch size가 존재한다.

**이게 얼마나 문제가 되는가?** 구체적인 예로, TPU v5p에서 BS 2M 토큰으로 LLaMA-3 70B를 학습하고 싶다고 하자. LLaMA-3 70B는 $F\approx 30,000$이다. 위 절들로부터 다음을 안다:

* Tensor Parallelism은 $Y = M_Y \cdot F / 2550 \approx 11 \cdot M_Y$까지 할 수 있다.
* FSDP는 $B / N > 2550 / M_X$인 한 할 수 있다. 즉 BS=2M과 3개 axis의 data parallelism으로 학습한다면 최대 $\approx 2400$칩, 대략 TPU v5p pod의 1/4까지만 쓸 수 있다.
* FSDP + Tensor Parallelism을 조합하면 $B / N < 2550^2 / (2 \cdot 30000) = 108$일 때 comms-bound가 되므로, 대략 18k칩까지 스케일할 수 있다! 하지만 TPU v5p pod의 최대 크기는 8k칩이므로, 그 너머로는 DCN을 써야 한다.

요약하면, BS=1M으로는 대략 X (FSDP) = 1024, Y (TP) = 8을 쓰는 좋은 학습 레시피가 있지만, BS=2M이면 DCN을 써야 한다. 위에서 언급했듯 DCN arithmetic intensity는 $\text{73,440}$이므로, ICI 도메인당 batch size가 이보다 크도록만 하면 된다. 이건 우리에게 식은 죽 먹기다. pod 2개면 pod당 BS가 1M이고 TPU당 batch size가 111이니 훌륭하다(약간 아슬아슬할 수는 있지만 이론적으로는 건전하다).

<div class="takeaway">

**요점(Takeaway):** pod당 batch size가 최소 73k 토큰이기만 하면, 순수 data parallelism을 써서 여러 TPU pod에 걸쳐 스케일하는 것은 꽤 간단하다.

</div>

## TPU에서의 LLM 학습 요점 정리

* 병렬화를 늘리거나 batch size를 줄이면 칩당 수행되는 연산량이 줄어들기 때문에, 둘 다 우리를 더 communication-bound로 만드는 경향이 있다.

* 적당한 context 길이(~32k)까지는 Transformer를 MLP 블록의 스택으로 모델링해도 무방하며, 여러 병렬화 전략 각각을 레이어당 두세 개의 주요 matmul을 어떻게 shard하는지로 정의할 수 있다.

* 학습 중에 고려하는 주요 병렬화 전략은 4가지이며, 각각 고유한 bandwidth·연산 요구 조건을 가진다(data parallelism, FSDP, tensor parallelism, 혼합 FSDP + tensor parallelism).

| **전략**                                 | **설명**                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data Parallelism**                         | activation은 batch로 shard되고, 나머지는 전부 완전히 복제되며, backward pass에서 gradient를 all-reduce한다.                                                                      |
| **FSDP**                                     | activation·weight·optimizer가 batch로 shard되고, weight는 사용 직전에 gather되며, gradient는 reduce-scatter된다.                                                               |
| **Tensor Parallelism (Megatron, Model이라고도 부름)** | activation은 $$d_\text{model}$$을 따라, weight는 $$d_{ff}$$를 따라 shard된다. activation은 W<sub>in</sub> 전에 gather되고, 결과는 W<sub>out</sub> 후에 reduce-scatter된다. |
| **혼합 FSDP + Tensor Parallelism**          | 위 둘의 조합으로, FSDP가 model-shard된 weight를 gather한다.                                                                                                                          |

각 방법의 "공식"은 다음과 같다:

$$
\small
\begin{array}{cc}
\text{Strategy} & \text{Formula}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}
$$

* 각 전략에는 디바이스당 연산과 통신에 따라 네트워크/통신 bound가 되는 한계가 있다. 다음은 $$X$$가 FSDP, $$Y$$가 tensor parallelism이라고 가정할 때의 레이어당 연산량과 통신량이다.

$$
\small
\begin{array}{ccc}
\text{Strategy} & \text{Compute per layer} & \text{Comms per layer} \\
& \text{(ignoring gating einsum)} & \text{(bytes, forward + backward pass)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}
$$

* 순수 data parallelism은 모델과 optimizer state가 파라미터 수의 10배 바이트를 쓰기 때문에 유용한 경우가 드물다. 즉 몇 B 파라미터 이상을 메모리에 담을 수 있는 경우가 드물다.

* Data parallelism과 FSDP는 $$\text{batch size per shard} < C / W$$, 즉 네트워크의 arithmetic intensity보다 작아지면 comms bound가 된다. ICI에서 이 값은 2,550이고 DCN에서는 약 71,000이다. 병렬 axis를 늘리면 이 값을 키울 수 있다.

* Tensor parallelism은 $$\lvert Y\rvert > F / 2550$$이면 comms bound가 된다. **대부분의 모델에서 8~16-way 정도다.** 이는 batch size와 무관하다.

* 혼합 FSDP + tensor parallelism은 batch size를 $$2550^2 / 2F \approx 100$$까지 낮출 수 있게 해 준다. 놀랄 만큼 낮은 값이다.

* pod들에 걸친 data parallelism은 DCN-bound가 되지 않으려면 pod당 최소 batch size가 대략 71,000이어야 한다.

* 요컨대 batch size가 크거나 모델이 작으면 만사가 단순하다. data parallelism을 하거나, DCN에 걸쳐 FSDP + data parallelism을 하면 된다. 흥미로워지는 것은 그 중간 구간이다.

## 연습 문제

이 절에서는 LLaMA-2 13B를 기본 모델로 쓰자. 모델 세부 사항은 다음과 같다:

| 하이퍼파라미터 | 값  |
| ---------- | ------ |
| L          | 40     |
| D          | 5,120  |
| F          | 13824  |
| N          | 40     |
| K          | 40     |
| H          | 128    |
| V          | 32,000 |

LLaMA-2는 별도의 embedding·출력 행렬과 gated MLP 블록을 가진다.

**문제 1:** LLaMA-2 13B의 파라미터는 몇 개인가(우습게 들리겠지만 직접 계산해 보라)? *[Transformer 수학](/scaling-book/transformers/)에서처럼 LLaMA-3는 3개의 큰 FFW 행렬 — up-projection 2개와 down-projection 1개 — 을 가진다는 점에 유의하라. 이 장에서는 두 "gating" einsum 행렬을 무시했지만, 이들은 이 장의 W<sub>in</sub>과 동일하게 행동한다.*

<details>
<summary>정답 보기</summary>

* FFW 파라미터: $$3LDF$$ = `8.5e9`
* Attention 파라미터: $$4DNHL$$ = `4.2e9`
* Vocabulary 파라미터: $$2VD$$ = `0.33e9`
* 합계: `8.5e9 + 4.2e9 + 0.33e9 = 13.0e9`, 예상대로다!

</details>

**문제 2:** BS=16M 토큰으로 Adam을 써서 학습한다고 하자. 병렬화는 잠시 잊고, 모델의 파라미터·optimizer state·activation이 쓰는 총 메모리는 얼마인가? *파라미터는 bf16, optimizer state는 fp32로 저장하고, 레이어당 세 번(세 개의 큰 matmul 뒤에서) activation을 checkpoint한다고 가정하라.*

<details>
<summary>정답 보기</summary>

파라미터(bf16)와 두 optimizer state(fp32, 1차·2차 모멘트 accumulator)에 쓰이는 총 메모리는 `(2 + 4 + 4) * 13e9 ~ 130GB`다. 처음 두 matmul 뒤의 activation은 shape이 $BF$이고 마지막 matmul 뒤는 $BD$이므로(위 Transformer 다이어그램 기준), bf16 기준 총 메모리는 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$, 즉 `B=16e6`이므로 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`다. 나머지 activation은 대체로 무시해도 될 수준이다.

</details>

**문제 3:** 32k 시퀀스 길이와 총 batch size 3M 토큰으로 TPUv5p 16x16x16 슬라이스에서 학습하고 싶다고 하자. 위와 같이 bfloat16 weight와 float32 optimizer를 쓴다고 가정한다.

1. 순수 data parallelism을 쓸 수 있는가? 그 이유는?
2. 순수 FSDP를 쓸 수 있는가? 그 이유는? 순수 FSDP라면 디바이스당 메모리는 얼마나 쓰게 되는가(gradient checkpointing은 3개의 큰 FFW matmul 뒤에서만 한다고 가정)?
3. 혼합 FSDP + tensor parallelism을 쓸 수 있는가? 그 이유는? 가능하다면 $X$와 $Y$는 얼마여야 하는가? 디바이스당 저장되는 메모리는 얼마인가? roofline FLOPs 추정만 쓰고 attention을 무시하면, 40% MFU에서 학습 스텝 하나는 얼마나 걸리는가?

<details>
<summary>정답 보기</summary>

우선 수치 몇 개를 적어 보자. 32k 시퀀스 길이와 3M batch size면 시퀀스 단위 batch size는 96이다. TPU v5p 16x16x16 슬라이스에는 `393TB`의 HBM이 있다.

1. 순수 data parallelism은 쓸 수 없다. 파라미터와 optimizer state를 각 칩에 복제하는데, 이것만 이미 (Q2에서) 약 130GB로 칩당 HBM(96GB)보다 크기 때문이다.

2. 우선 메모리만 보자. Q2에서 BS=16M을 3M으로 바꾸면 checkpoint activation 총량은 `~7.86e12`가 되고, optimizer state 1.3e11을 더하면 거의 정확히 8e12 = 8TB다. TPUv5p 슬라이스에는 총 `393TB`의 HBM이 있으므로 HBM 한도는 안전하게 밑돈다. 다음으로 comms-bound인지 compute-bound인지 보자. 4096칩과 3개 axis의 병렬화면 최소 batch size는 `850 * 4096 = 3.48M` 토큰이다. 우리의 3M batch size보다 약간 크다. 즉 실제로는 comms-bound가 되고 만다. 슬프다. 따라서 일반적인 답은 **아니오, FSDP만으로는 안 된다**.

3. 이제 주된 걱정이 comms-bound라는 것을 알았으니 수치를 넣어 보자. 우선 위로부터, 혼합 FSDP + tensor parallelism의 칩당 batch size는 여기서 $2550^2 / 2F = 235$보다 커야 한다. 즉 이론상 가능하다! 각각을 얼마나 할지 알아보자.

$X_{opt} = \sqrt{(B / F) \cdot (M_X / M_Y) \cdot N}$ 규칙이 있으므로, 여기서는 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`이 되어 대략 1024-way DP와 4-way TP를 하게 된다. TPU당 메모리는 (2)와 같고, 스텝 시간은 그냥 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`다.

</details>

<div class="takeaway">

**5부는 여기까지!** 이 내용을 실제 LLaMA 모델에 적용하는 6부는 [여기](/scaling-book/applied-training/)에서 볼 수 있다!

</div>

## 부록

### 부록 A: backward pass 통신량 유도하기

위에서 Transformer 레이어의 forward pass를 Out[B, D] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F] \*<sub>F</sub> W<sub>out</sub>[F, D]로 단순화했다. backward pass에 필요한 통신은 어떻게 유도할까?

이는 앞 장에서 본 단일 matmul **Y = X * A**에 대한 규칙에서 꽤 자연스럽게 따라 나온다:

$$
\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)
$$

$$
\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T
$$

이를 이용하면 다음 공식들을 얻는다(Tmp[B, F]는 In[B, D] \* W<sub>in</sub>[D, F]를 나타낸다):

<div class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] \*<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub>[D, F] = In[B, D] \*<sub>B</sub> dTmp[B, F]
4. dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F]

</div>

이 공식들은 수학적 진술이며 sharding에 대한 언급이 전혀 없다는 점에 유의하라. backward pass가 할 일은 이 네 값을 계산하는 것이다. 따라서 필요한 통신을 알아내려면, 위 네 식에서 matmul될 값들(Tmp, dOut, W<sub>out</sub>, W<sub>in</sub>)의 sharding — 우리의 병렬화 방식이 지정한다 — 을 가져다가, sharded matmul의 규칙을 적용해 어떤 통신을 해야 하는지 알아내면 된다. dOut은 Out과 같은 방식으로 shard된다는 점에 유의하라.

[^1]: 여기서는 통신 한계에 집중한다 — 메모리 용량 제약도 중요하지만, pre-training에서 rematerialization(activation checkpointing)과 아주 많은 수의 칩을 쓸 때는 보통 우리를 제약하지 않기 때문이다. 또한 MoE를 위한 expert parallelism은 설계 공간을 크게 넓히므로 여기서는 다루지 않고, dense Transformer라는 기본 경우만 다룬다.
[^2]: Adam은 파라미터와 1차·2차 accumulator를 저장한다. 파라미터는 bfloat16이고 optimizer state는 float32이므로 파라미터당 `2 + 8 = 10` 바이트가 된다.
[^3]: 이는 gradient checkpoint를 포함하지 않으므로 실제로 유용한 수치는 아니다. batch가 1 토큰일 때의 절대적 하한이다.
[^4]: 이 분할이 ICI mesh 위에서 이루어진다고 가정하므로, 관련된 네트워크 bandwidth는 $W_\text{ici}$다.
[^5]: 엄밀히는 FSDP가 순수 DP에는 없는 통신을 forward pass에 더하지만, 이는 backward pass와 같은 비율이므로 통신 roofline에는 영향이 없다. 핵심은 ZeRO-3가 backward pass의 AllReduce를 AllGather와 ReduceScatter로 바꾸는데, 이 둘의 총 통신량이 AllReduce와 같다는 것이다.
