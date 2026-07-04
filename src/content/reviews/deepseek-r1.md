---
title: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning'
arxivId: '2501.12948'
authors: 'Guo et al. (DeepSeek-AI)'
lab: 'DeepSeek-AI'
venue: 'Nature 2025'
date: 2026-07-04
tags: [post-training, efficiency]
topic: 'post-training'
summary: 'Pure RL with rule-verifiable rewards (GRPO) incentivizes o1-level reasoning to emerge from a base model without supervised CoT, and the capability transfers to small dense models via distillation.'
summary_ko: 'Rule-verifiable reward 기반 pure RL(GRPO)만으로 base model에서 o1급 reasoning이 emergent하게 나타남을 보이고, 그 능력을 distillation으로 small dense model까지 이전한 open recipe.'
links: [minimax-01, mobilellm]
resources:
  - { label: 'arXiv', url: 'https://arxiv.org/abs/2501.12948' }
  - { label: 'PDF', url: 'https://arxiv.org/pdf/2501.12948' }
  - { label: 'GitHub', url: 'https://github.com/deepseek-ai/DeepSeek-R1' }
  - { label: 'HuggingFace', url: 'https://huggingface.co/deepseek-ai/DeepSeek-R1' }
analysis:
  ko:
    background: "OpenAI o1이 inference-time long chain-of-thought 스케일링으로 큰 reasoning 도약을 보여줬지만 training recipe는 비공개였고, open 진영은 human-annotated CoT에 대한 SFT에 의존하고 있었다."
    problem: "Supervised reasoning demonstration 없이 base model에 pure RL만으로 강한 reasoning을 유도할 수 있는지, 그리고 그것을 실사용 가능한 general model로 만드는 방법이 미해결이었다."
    prior_limits: "기존 공개 시도들 — process reward model, MCTS 계열 search, 대규모 human CoT annotation — 은 reward hacking, search space 폭발, annotation 비용에 막혀 o1급 general reasoning에 도달하지 못했다."
    goal: "Open recipe로 o1급 reasoning을 재현한다 — pure RL의 R1-Zero로 메커니즘을 입증하고, multi-stage pipeline의 R1로 실사용성을 확보하고, distillation으로 small dense model까지 능력을 이전한다."
    method: "DeepSeek-V3-Base(671B MoE, 37B activated)에 critic 없이 group 평균 reward를 baseline으로 쓰는 GRPO를 rule-based accuracy·format(think 태그) reward로 학습해 R1-Zero를 얻고, R1은 long-CoT cold-start SFT → language-consistency reward를 더한 reasoning-oriented RL → rejection sampling 기반 약 800k 샘플 SFT(reasoning 600k + non-reasoning 200k) → all-scenario RL의 4단 pipeline로 훈련한다."
    key_idea: "정답을 rule로 검증 가능한 reward(accuracy + format)만으로도 충분한 training signal이 되어, neural reward model도 reasoning SFT도 없이 self-verification·reflection 같은 reasoning 행동이 RL incentive에서 emergent하게 나타난다."
    validation: "AIME 2024, MATH-500, CNMO 2024, GPQA Diamond, MMLU/-Redux/-Pro, Codeforces, LiveCodeBench, SWE-bench Verified, Aider와 open-ended judge(AlpacaEval 2.0, ArenaHard)에서 평가하고, Qwen2.5/Llama3 기반 1.5B–70B distilled 모델 6종은 reasoning 벤치 부분집합으로 검증했다."
    results: "R1-Zero는 RL만으로 AIME 2024 pass@1을 **15.6%→71.0%**(cons@64 **86.7%**)로 끌어올리고, R1은 AIME **79.8%**, MATH-500 **97.3%**, GPQA Diamond **71.5%**, Codeforces rating **2,029**(96.3 percentile)로 OpenAI-o1-1217과 동급이다."
    comparison: "OpenAI-o1-1217, o1-mini, GPT-4o-0513, Claude-3.5-Sonnet-1022, DeepSeek-V3, QwQ-32B-Preview와 직접 비교했으며, 특히 distilled R1-Qwen-32B(AIME **72.6%**)가 같은 32B base에 직접 RL을 적용한 경우(**47.0%**)를 크게 앞서 대형 teacher distillation이 소형 모델 직접 RL보다 낫다는 것을 보였다."
    significance: "o1급 reasoning recipe를 처음으로 open-source화해 verifiable-reward RL(RLVR)+GRPO를 post-training의 표준 패러다임으로 만들었고, 800k 샘플 distillation으로 small dense model도 reasoning을 물려받을 수 있음을 증명했다."
    limitations: "논문 명시 한계로 function calling·multi-turn·JSON output에서 DeepSeek-V3보다 약하고, 중국어·영어 외 언어에서 language mixing이 발생하며, few-shot prompt에 민감하고, 평가 지연 때문에 software engineering에는 대규모 RL을 적용하지 못했다 — 또한 수천 토큰의 long-CoT decode는 latency와 KV cache 압박이라는 실질적 serving 비용을 만든다(리뷰어 판단)."
    future_work: "논문은 long CoT의 general capability 확장, 다국어 mixing 해결, software engineering용 asynchronous-evaluation RL을 제시하며, rule-verifiable domain 밖의 reward 설계가 더 넓은 열린 문제로 남아 있다(리뷰어 판단)."
    resources: "R1-Zero, R1과 distilled 모델 6종(Qwen2.5/Llama3, 1.5B–70B)의 weight가 MIT license로 HuggingFace에 공개됐지만, RL training code와 학습 데이터는 비공개다."
  en:
    background: "OpenAI o1 showed that scaling inference-time chain-of-thought yields large reasoning gains, but its training recipe stayed closed and the open community was still relying on SFT over human-annotated CoT."
    problem: "It was unresolved whether strong reasoning can be incentivized by pure RL on a base model — with no supervised reasoning demonstrations at all — and how to turn that into a usable general-purpose model."
    prior_limits: "Prior open attempts — process reward models, MCTS-style search, and large-scale human CoT annotation — ran into reward hacking, search-space explosion, and annotation cost, and none reached o1-level general reasoning."
    goal: "Reproduce o1-class reasoning with an open recipe: prove the mechanism with pure-RL R1-Zero, make it usable via the multi-stage R1 pipeline, and transfer the capability to small dense models through distillation."
    method: "Starting from DeepSeek-V3-Base (671B MoE, 37B activated), GRPO — a critic-free policy gradient that baselines each sample against its group's mean reward — is trained with rule-based accuracy and think-tag format rewards to produce R1-Zero, and R1 adds a four-stage pipeline: long-CoT cold-start SFT, reasoning-oriented RL with a language-consistency reward, rejection-sampling SFT on ~800k samples (600k reasoning + 200k non-reasoning), and a final all-scenario RL stage."
    key_idea: "Rule-verifiable rewards (answer correctness + output format) alone are sufficient training signal — self-verification, reflection, and long deliberation emerge from RL incentives without any neural reward model or reasoning SFT."
    validation: "Evaluated on AIME 2024, MATH-500, CNMO 2024, GPQA Diamond, MMLU/-Redux/-Pro, Codeforces, LiveCodeBench, SWE-bench Verified, Aider, and open-ended judges (AlpacaEval 2.0, ArenaHard), with six distilled 1.5B–70B Qwen2.5/Llama3 checkpoints tested on the reasoning subset."
    results: "R1-Zero lifts AIME 2024 pass@1 from **15.6% to 71.0%** (**86.7%** cons@64) by RL alone, and R1 reaches **79.8%** on AIME, **97.3%** on MATH-500, **71.5%** on GPQA Diamond, and a **2,029** Codeforces rating (96.3 percentile), on par with OpenAI-o1-1217."
    comparison: "Benchmarked head-to-head against OpenAI-o1-1217, o1-mini, GPT-4o-0513, Claude-3.5-Sonnet-1022, DeepSeek-V3, and QwQ-32B-Preview; notably, distilled R1-Qwen-32B (**72.6%** AIME) far exceeds direct RL on the same 32B base (**47.0%**), showing distillation from a strong teacher beats small-scale RL."
    significance: "It open-sourced the first o1-class reasoning recipe, establishing verifiable-reward RL (RLVR) with GRPO as the default post-training paradigm and proving via 800k-sample distillation that small dense models can inherit reasoning."
    limitations: "The paper states R1 trails DeepSeek-V3 on function calling, multi-turn, and JSON output, mixes languages outside Chinese/English, is sensitive to few-shot prompting, and lacked large-scale RL on software engineering due to evaluation latency — and the thousands-of-tokens long-CoT decode is itself a real serving cost in latency and KV-cache pressure (reviewer judgment)."
    future_work: "Stated directions are extending long CoT to general capabilities, fixing multilingual mixing, and asynchronous-evaluation RL for software engineering; the broader open problem is reward design beyond rule-verifiable domains (reviewer judgment)."
    resources: "Weights for R1-Zero, R1, and six distilled Qwen2.5/Llama3 models (1.5B–70B) are released under the MIT license on HuggingFace, but the RL training code and training data are not public."
source: 'manual'
rating: 5
---

## Notes

DeepSeek-R1 is the report that moved reasoning post-training from closed labs into the open: R1-Zero is the cleanest existence proof that rule-verifiable rewards alone can elicit long-horizon reasoning from a base model, and the four-stage R1 pipeline (cold-start SFT → RLVR → rejection-sampling SFT → alignment RL) became the template most open reasoning models now follow. The negative results are as instructive as the headline — neural PRMs collapsed to reward hacking and MCTS to search-space explosion, which is precisely why the recipe reduces to outcome-verifiable rewards. From an efficiency standpoint its two lasting artifacts are the distilled 1.5B–70B dense models, now a standard testbed for compression and serving work, and the shift of inference cost into long thinking traces — the very regime that makes long-generation KV-cache research matter. 이후 공개된 reasoning 모델 대부분이 이 recipe를 변형해 쓰고 있고, GRPO/RLVR 계열 후속 연구의 출발점이 된 논문이다.
