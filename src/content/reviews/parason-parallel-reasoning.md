---
title: "Parason: Revealing Subtask and Trial Parallelism in LLM Reasoning"
arxivId: "2608.24658"
date: 2026-08-27
tags: ["llm-reasoning", "test-time-scaling", "parallel-decoding", "post-training"]
topic: 'serving'
summary: "Parason shows that most of the hidden parallelism in LLM reasoning traces is Trial Parallelism (parallel speculative attempts), not the Subtask Parallelism prior systems target, and trains models via PA-GRPO plus tool-call execution for a real 1.7x wall-clock speedup."
summary_ko: 'Parason은 LLM 추론 과정에 숨어 있는 병렬성의 대부분이 기존 시스템이 다루던 Subtask Parallelism이 아니라 Trial Parallelism(병렬 추측 시도)임을 보이고, PA-GRPO 학습과 도구 호출 실행을 결합해 실제 1.7배의 지연시간 단축을 달성한다.'
links: ['deepseek-r1']
resources:
  - label: 'arXiv abstract'
    url: 'https://arxiv.org/abs/2608.24658'
  - label: 'Project website'
    url: 'https://zhengyangzhang06.github.io/parason-web/'
figures:
  - src: /figures/parason-parallel-reasoning/fig1.png
    caption: "Subtask Parallelism (every branch required, AND-relation) vs. Trial Parallelism (competing speculative attempts, OR-relation) -- the taxonomy split the paper reveals."
    caption_ko: "Subtask Parallelism(모든 분기가 필요한 AND 관계)과 Trial Parallelism(경쟁하는 병렬 추측 시도, OR 관계) -- 논문이 밝힌 두 병렬성 유형의 구분."
    credit: "Figure 1 from arXiv:2608.24658 -- authors figure"

analysis:  # per key: 2-4 sentences — first = one crisp information-dense claim; add an apt analogy where it genuinely clarifies
  ko:
    background: 'test-time 추론 스케일링은 LLM 문제 해결 능력을 크게 끌어올렸지만, 표준 autoregressive 디코딩은 긴 추론 과정을 여전히 토큰 단위로 순차 실행한다. 그래서 어려운 문제는 실제 잠재된 병렬성이 있음에도 며칠에서 몇 주에 이르는 wall-clock 시간이 걸릴 수 있다.'
    problem: '기존 병렬 추론 시스템 대부분은 하나의 과제를 독립적으로 풀 수 있는 하위 문제로 나누는 Subtask Parallelism에만 집중해 왔다. 추론 과정 안에는 더 크지만 거의 다뤄지지 않은 두 번째 병렬성 원천이 남아 있다.'
    prior_limits: '기존 시스템들은 subtask 분해 능력만 학습·평가하기 때문에, 이 논문이 직접 측정한 사실을 놓친다: DeepSeek-V4의 실제 추론 과정을 HLE 벤치마크에서 분석하면, 여러 추측 시도를 나란히 실행하고 결과를 종합하는 Trial Parallelism이 병렬화 가능한 추론 연산의 **65.5%**를 차지하며, DeepSeek-R1과 DeepSeek-V4 모두에서 58%를 넘는다. 문제가 어려울수록 이 비중은 더 커진다.'
    goal: '하나의 추론 과정 안에 있는 두 병렬성 형태를 모두 드러내고, 주어진 문제에 실제로 어떤 형태(또는 조합)가 필요한지를 모델이 스스로 판단하도록 학습시킨 다음, 그 판단을 이론적 FLOP 절감이 아니라 실제로 측정된 wall-clock 가속으로 이어지게 하는 것이 목표다.'
    method: 'Parason은 먼저 기존의 순차적 추론 과정을 문맥 자유 문법(context-free grammar)을 이용해 구조화된 병렬 궤적으로 변환한다. 이 문법은 모든 분기가 필요한 subtask 분기(AND 관계)와, 서로 경쟁하는 대안인 trial 분기(OR 관계, 이후 문맥에는 모든 분기 내용이 이어붙여져 최종 종합에 쓰인다)를 모두 표현할 수 있다. 그런 다음 정확도·지연시간·두 병렬성 비율을 함께 반영하는 보상 함수를 쓰는 Parallelism-Aware Group Relative Policy Optimization(PA-GRPO)으로 학습한다.'
    key_idea: '핵심은 고정된 분해 휴리스틱을 손으로 설계하거나 학습되지 않은 구조의 단순 best-of-N 샘플링에 의존하는 대신, 실제 지연시간을 반영한 보상을 통해 모델 스스로가 언제 subtask를 나누고 언제 trial을 띄울지 학습하게 만드는 것이다. 비유하자면, 모든 문제에 고정된 조직도를 적용하는 것이 아니라, 업무를 전문가들에게 나눠 맡길지(subtask) 아니면 여러 사람이 같은 답을 향해 경쟁하게 하고 그중 최선을 고를지(trial)를 사안별로 학습하는 매니저를 훈련시키는 것에 가깝다.'
    validation: '학습된 병렬 구조는 추론 시점에 도구 호출(tool call)을 통해 실제로 실행되며, 이것이 학습된 병렬성을 측정된 wall-clock 절감으로 바꾸는 단계다. 병렬 구조에 보상을 주더라도 추론 시점에 순차적으로 재생되는 정책이라면 실제 가속은 전혀 없을 것이라고 논문은 명확히 밝히는데, 그래서 이 실행 경로를 방법론의 부수 요소가 아니라 핵심 구성 요소로 다룬다.'
    results: 'AIME24에서 2,048 토큰의 고정 사고 예산 기준으로 PA-GRPO는 **34.7%**의 정확도를 기록해 동일 예산의 SFT-only 기준선 **16.8%**를 크게 앞선다. 8,192 토큰에서는 **60.3% 대 41.8%**다. AIME24, AIME25, AMC, Math500 네 개 벤치마크 평균에서 Parason은 **84.7%** 정확도(AIME25 70.6%, AMC 97.5%, Math500 94.6%)를 달성하면서 평균 **약 1.7배**의 wall-clock 가속을 제공한다.'
    comparison: '4개 벤치마크 평균 정확도 81.0%, 토큰 지연시간 14.8k를 기록한 기존 병렬 추론 시스템 ThreadWeaver와 비교하면, Parason은 더 짧은 critical path에서 더 높은 정확도를 낸다 -- 이는 단순히 병렬 연산을 더 쓴 것이 아니라 trial-vs-subtask 분류 자체가 실질적인 역할을 한다는 증거다. 65.5%라는 Trial Parallelism 측정치 자체도 하나의 비교 기준이 된다: subtask 분해에만 집중했던 기존 시스템들이 어려운 문제에서 활용 가능한 병렬성의 더 작은 몫만 노려 왔음을 보여준다.'
    significance: 'test-time-scaling 효율을 순수한 디코딩 엔진 문제가 아니라 분류(taxonomy) 문제로 재구성한다 -- 달성 가능한 가속의 상한은 주어진 추론 과정이 얼마나 subtask 형태인지 trial 형태인지에 달려 있고, 이 비율은 난이도에 따라 달라진다. 마치 교통 정체를 진단할 때, 지연의 대부분이 좁은 도로(subtask 부족) 때문이 아니라 교차로에서 대기하는 차들(trial 낭비) 때문이라면 도로만 넓혀서는 해결되지 않는 것과 비슷하다. 추론 효율 연구 전반에 대한 더 넓은 교훈은, 가정된 분해 구조를 바탕으로 시스템을 만들기 전에 실제 연산의 구조부터 측정하라는 것이다.'
    limitations: '저자들이 밝힌 한계: 학습과 평가가 주로 수학 추론에 집중되어 있어, 동일한 taxonomy, 데이터 큐레이션 파이프라인, PA-GRPO 목적함수가 실제 에이전트 같은 다른 도메인으로 전이되는지는 불분명하다. 실험도 8B 규모 모델에 한정되어 있어, 다른 모델 계열이나 더 큰 규모에서도 같은 병렬성 패턴과 지연시간 이득이 유지되는지는 확인되지 않았다.'
    future_work: '저자들은 Parason을 더 다양한 백본 모델과 더 큰 규모로 확장해, 이 논문에서 관찰된 병렬성 패턴과 지연시간 이득이 8B 세팅을 넘어서도 일반화되는지 확인하는 것을 향후 과제로 명시한다.'
    resources: '논문은 프로젝트 웹사이트(접속 확인됨)와 Hugging Face 데이터셋 페이지를 링크하지만, 논문에 적힌 GitHub 코드 링크(parason-internal)는 404를 반환해 공개 여부를 확인할 수 없었고, 데이터셋 페이지도 인증이 필요해 공개 접근 여부를 확인하지 못했다.'
  en:
    background: 'Test-time reasoning scaling has substantially improved LLM problem-solving, but standard autoregressive decoding still executes long reasoning traces strictly token by token, so the hardest problems can take days to weeks of wall-clock time even though the underlying computation has real latent parallelism.'
    problem: 'Prior parallel-reasoning systems concentrate almost entirely on Subtask Parallelism -- decomposing a task into independent sub-problems the model can solve concurrently. A second, larger source of exploitable parallelism inside reasoning traces has gone essentially unaddressed.'
    prior_limits: 'Because those systems train and evaluate only the subtask-decomposition skill, they miss what this paper measures directly: on DeepSeek-V4 reasoning traces on the HLE benchmark, Trial Parallelism -- running multiple speculative solution attempts side by side and aggregating them -- makes up **65.5%** of all parallelizable reasoning computation, exceeding 58% for both DeepSeek-R1 and DeepSeek-V4 across the benchmarks studied, and its share grows further as problems get harder.'
    goal: 'Reveal both forms of parallelism inside a single reasoning trace, then train one model to decide, case by case, which form (or combination) a given problem actually calls for -- and carry that decision through to a real, measured wall-clock speedup rather than a theoretical FLOP-count estimate.'
    method: 'Parason first rewrites existing sequential reasoning traces into structured parallel trajectories using a context-free grammar that can express both subtask branches (AND-relation, every branch needed for the final answer) and trial branches (OR-relation, competing alternatives whose content is still concatenated into the following context for synthesis). It then trains on these structured trajectories with Parallelism-Aware Group Relative Policy Optimization (PA-GRPO), whose reward jointly balances task accuracy, latency, and the two parallelism ratios.'
    key_idea: 'The core move is to let the model itself learn when to fork a subtask versus launch a trial, priced against real latency, rather than hand-coding a fixed decomposition heuristic or relying on plain best-of-N sampling with no learned structure. An apt analogy: this is less like assigning a fixed org chart to every problem, and more like training a manager who learns, case by case, whether to split work among specialists (subtasks) or have several people race toward the same answer and pick the best one (trials).'
    validation: 'The learned parallel structure is executed at inference time through tool calls -- the step that actually converts trained parallelism into measured wall-clock savings. The paper states plainly that a policy rewarded for parallel structure but replayed serially at inference would show no real speedup at all, which is why this execution path is treated as a first-class part of the method rather than an implementation detail.'
    results: 'On AIME24, at a fixed 2,048-token thinking budget, PA-GRPO reaches **34.7%** accuracy versus **16.8%** for an SFT-only baseline at the same budget; at 8,192 tokens the gap is **60.3% vs. 41.8%**. Across four reasoning benchmarks (AIME24, AIME25, AMC, Math500), Parason reaches an average accuracy of **84.7%** (70.6% AIME25, 97.5% AMC, 94.6% Math500) while delivering an average **~1.7x** wall-clock acceleration.'
    comparison: 'Against ThreadWeaver, a prior parallel-reasoning system that reaches 81.0% four-benchmark average accuracy at 14.8k token latency, Parason reaches higher accuracy at a shorter critical path -- evidence the trial-versus-subtask taxonomy, not just more parallel compute, is doing real work. The 65.5% Trial-Parallelism measurement is itself a comparison point against every prior subtask-only system: it shows they were targeting the smaller share of a hard problem exploitable parallelism.'
    significance: 'This reframes test-time-scaling efficiency as a taxonomy problem rather than purely a decoding-engine problem -- how much speedup is achievable depends on how much of a given trace is subtask-shaped versus trial-shaped, and that ratio shifts with difficulty. It is a bit like diagnosing a traffic jam: if most of the delay comes from cars idling at intersections (trials) rather than narrow roads (subtasks), widening the roads alone will not help -- fixing the wrong cause wastes the effort. The broader lesson for inference-efficiency work generally: measure the actual structure of the computation you are trying to parallelize before building a system around an assumed decomposition.'
    limitations: 'Paper-stated: training and evaluation focus mainly on mathematical reasoning, so it is unclear whether the same taxonomy, data-curation pipeline, and PA-GRPO objective transfer to other domains such as real-world agents. Experiments are also limited to 8B-scale models -- a controlled testbed that does not show whether the same parallelism patterns and latency gains hold across model families and larger sizes.'
    future_work: 'The authors name scaling Parason to more backbone models and larger sizes as explicit future work, to test whether the observed parallelism patterns and latency gains generalize beyond the 8B setting studied here.'
    resources: 'The paper links a project website (verified live) and a Hugging Face dataset page; the GitHub code link named in the paper (parason-internal) returns a 404 and could not be verified as public, and the dataset page requires authentication that could not be confirmed as open access -- so full reproducibility could not be independently verified beyond the paper and website.'
thread:
  ko: |-
    test-time 스케일링이 LLM 추론 능력을 끌어올리면서, 그 계산을 병렬화해 지연시간을 줄이려는 시도가 이어져 왔다. 독립 병렬 샘플링과 다수결 투표(majority voting)가 가장 단순한 형태였고, 이후 구조화되고 적응적인 병렬 추론 시스템들(예: ThreadWeaver)이 등장해 과제를 하위 문제로 나누는 Subtask Parallelism을 체계적으로 학습시키는 방향으로 발전했다.

    Parason이 만드는 전환은, 추론 과정에 숨어 있는 병렬성을 하나가 아니라 두 종류로 나눠 본다는 데 있다. Subtask Parallelism 옆에 Trial Parallelism -- 여러 추측 시도를 나란히 실행하고 종합하는 방식 -- 을 나란히 두고, DeepSeek 계열 모델의 실제 추론 과정에서 어느 쪽이 실제로 더 큰 비중을 차지하는지 직접 측정한다. 그 결과가 65.5%라는, 기존 시스템들이 놓치고 있던 압도적인 Trial 비중이다. 여기서 그치지 않고 PA-GRPO라는 강화학습으로 모델 스스로 두 형태를 구분해 활용하도록 학습시키고, 도구 호출을 통한 실행으로 이론적 절감을 실제 wall-clock 가속으로 연결한다.

    이 논문이 여는 질문은 두 갈래다. 하나는 저자들이 직접 밝힌 것으로, 수학 추론을 넘어 실제 에이전트 워크플로우나 더 큰 모델에서도 같은 taxonomy와 65.5%라는 비율이 유지되는지다. 다른 하나는 더 일반적인 교훈으로, test-time-scaling을 다루는 다른 영역(코드 생성, 도구 사용 에이전트, 심지어 diffusion 기반 언어모델의 병렬 디코딩)에서도 "내가 병렬화하려는 계산의 실제 구조가 무엇인가"를 먼저 측정하지 않고 시스템을 설계하고 있지는 않은지 되묻게 한다.
  en: |-
    As test-time scaling has driven LLM reasoning gains, parallelizing that computation to cut latency has become its own line of work. Independent parallel sampling with majority voting was the simplest form, followed by structured and adaptive parallel-reasoning systems (e.g. ThreadWeaver) that systematically trained Subtask Parallelism -- decomposing a task into sub-problems solved concurrently.

    The shift Parason makes is to split the parallelism hidden in a reasoning trace into two kinds instead of one. Alongside Subtask Parallelism it places Trial Parallelism -- running multiple speculative attempts side by side and aggregating them -- and directly measures, on DeepSeek-family reasoning traces, which one actually dominates. The answer is 65.5%, an overwhelming Trial share that prior systems were missing entirely. From there, Parason trains a model with PA-GRPO reinforcement learning to tell the two forms apart and use them, then executes the result through tool calls so the theoretical savings become a real wall-clock speedup.

    What this opens splits into two threads. One is stated directly by the authors: whether the same taxonomy and the same roughly-65% Trial share hold beyond mathematical reasoning, in real agent workflows, and at model scales larger than 8B. The other is a broader lesson for anywhere test-time scaling is used -- code generation, tool-using agents, even parallel decoding in diffusion language models -- a prompt to ask whether a system is being designed around an assumed computation structure that was never actually measured.
sparks:
  - ko: '수학 추론을 넘어, 실제 에이전트 워크플로우(도구 호출·검색·다단계 계획)에서도 Trial Parallelism이 여전히 지배적인 비중을 차지할까? 저자들이 직접 명시한 열린 질문이다.'
    en: 'Beyond mathematical reasoning, does Trial Parallelism still dominate in real-world agent workflows (tool calls, search, multi-step planning)? An open question the authors name directly.'
  - ko: '8B 모델을 넘어 더 큰 백본으로 확장하면 같은 병렬성 패턴과 지연시간 이득이 유지될까, 아니면 모델 규모에 따라 subtask 대 trial 비중 자체가 달라질까? 저자들이 향후 과제로 명시한 방향이다.'
    en: 'Scaling beyond 8B backbones -- do the same parallelism patterns and latency gains hold, or does the subtask-versus-trial ratio itself shift with model scale? A direction the authors name as future work.'
source: "autosweep"
---

## Notes
