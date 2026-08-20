---
part: 11
title: "결론과 더 읽을거리"
title_en: "Conclusions and Further Reading"
original: "https://jax-ml.github.io/scaling-book/conclusion/"
summary: "책 전체를 마무리하며 감사의 말을 전하고, TPU·GPU 성능 엔지니어링과 LLM 스케일링을 더 깊이 공부할 수 있는 자료들을 소개한다."
date: 2026-08-20
published: true
---

> 끝까지 읽어 주어 고맙다! 여기서는 더 공부할 수 있는 참고 자료 몇 가지를 소개한다.

**책 전체를 다 읽어 주어 고맙고, 끝까지 완주한 것을 축하한다.** 마무리하기 전에 몇 가지 감사 인사를 전한다:

## 감사의 말

이 문서는 Google DeepMind의 많은 이들이 함께 쏟은 상당한 노력의 결과물이며, 그분들께 짧게나마 감사를 표하고 싶다!

- James Bradbury, Reiner Pope, Noam Shazeer, Blake Hechtman은 이 원고에 담긴 많은 아이디어를 처음 유도했고, Transformer를 시스템 관점에서 일찍부터 이해한 이들이다.
- Sholto Douglas는 이 문서의 첫 버전을 썼고 프로젝트를 시작시킨 장본인이다. 이 문서의 전체 서사는 누구보다 그의 공이다.
- Jacob Austin은 그 첫 버전을 거친 노트에서 더 다듬어지고 포괄적인 결과물로 바꾸는 작업을 이끌었다. 편집·포매팅·공개 작업의 대부분을 맡았고 다른 저자들의 기여를 조율했다.
- 그림과 애니메이션 대부분은 Anselm Levskaya와 Charlie Chen이 만들었다.
- Charlie Chen은 inference 섹션을 썼고 inference 그림 다수를 그렸다.
- Roy Frostig는 출간과 편집을 비롯해 여정의 많은 단계를 도왔다.

과정 전반에 걸쳐 중요한 피드백을 준 많은 분들께도 감사드린다. 특히 Zak Stone, Nikhil Sethi, Caitlin Stanton, Alek Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar Gupta, Jeff Dean, Corry Wang, Matt Johnson, Peter Hawkins를 비롯한 많은 분들이다. HTML 포매팅을 도와준 Ruiqi Gao에게도 감사드린다.

**모두 고맙습니다!**

<div class="takeaway">

떠나기 전에, NVIDIA GPU를 다루는 새 [12장](/scaling-book/gpus/)도 재미있게 읽을 수 있을 것이다!

</div>

## 더 읽을거리

관련된 글이 많이 있다. 다음과 같은 것들이다:

- [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html): 이 책의 정신을 이어받아 TPU 아키텍처를 깊이 있게 살펴보는 훌륭한 글.
- [**Domain specific architectures for AI inference**](https://fleetwood.dev/posts/domain-specific-architectures): 이 책의 정신과 맞닿아 있는 하드웨어·모델 심층 분석.
- [**A Domain-Specific Supercomputer for Training Deep Neural Networks**](https://dl.acm.org/doi/pdf/10.1145/3360307): 원조 TPU 논문 중 하나로, 여기서 다루지 않은 Google TPU 프로그램의 훌륭한 세부 사항이 많이 담겨 있다.
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html): LLM roofline과 성능 엔지니어링을 GPU와 PyTorch 중심으로 다룬 튜토리얼.
- [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): 갈수록 TPU 프로그래밍은 Pallas로 커스텀 kernel을 작성하는 일이 되어 가고 있다. 이 시리즈는 kernel 작성법과, 여기서 언급하지 않은 더 낮은 수준의 TPU 세부 사항을 다룬다.
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): GPU·CUDA에 특화된 글이긴 하지만, CUDA에서 matmul kernel을 최적화하는 과정을 보여주는 탁월한 블로그 글이다. TPU와 GPU가 어떻게 다른지 깊이 파보기에 좋은 자료다.
- [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): JAX의 병렬화 API를 다룬 정말 좋은 가이드로, 여기서 논의한 아이디어 일부를 실제로 구현하는 법을 배우기에 좋다.
- [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024): 전 동료 Rafi가 TPU 성능 엔지니어링에 관한 훌륭한 강의를 했는데, 슬라이드가 전부 GitHub에 있다. 이 책보다 여러 주제를 더 깊이 다룬다.
- [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): Transformer 추론의 수학을 상세히 다룬 논문. 이 문서의 많은 부분이 여기서 영감을 받았다.
- [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): 이 책의 GPU 버전이라 할 만한 자료로, PyTorch가 학습 중 병렬화 기법과 메모리 절약 기법을 어떻게 구현하는지 더 깊이 이야기한다.
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/): 이 책과 같은 아이디어를 다수 담은 블로그로, 일러스트가 훌륭하다.
- [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework): LLM 학습과 serving의 많은 세부 사항을 다루는 환상적인 Stanford 강의로, 유용한 연습 문제도 있다. 과제 1과 2가 특히 관련이 깊다.
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering): ML 인프라에 관한 대단히 실용적인 가이드로, 클라우드 업체와 협상하는 법, 클러스터 관리, GPU throughput 실측처럼 이 책에서 다루지 않은 주제들을 다룬다.
- [**ezyang's blog**](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/): PyTorch 리드가 sharding + PyTorch 전반을 다루는 블로그로, [PyTorch 내부 구조 가이드](https://blog.ezyang.com/2019/05/pytorch-internals/)와 [sharding된 행렬 곱셈 해설](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)이 있다. 그 밖에도 좋은 글이 많다.
- [**The Anatomy of Collective Communication**](https://www.aleksagordic.com/blog/collective-operations): 이 책의 정신에 맞게 GPU와 TPU의 collective를 잘 풀어낸 글. N-D collective와 GPU collective에 관해서는 이 책보다 더 나은 설명을 담고 있다.

이 분야에는 포괄적인 글이 들어설 여지가 아직 많이 남아 있으니, 이 원고가 그런 글을 더 많이 이끌어내는 계기가 되기를 바란다! 또한 우리는 이 분야가 공부하고 연구하기에 결실이 많은 영역이라고 믿는다. 많은 경우 하드웨어 가속기를 많이 갖고 있지 않아도 할 수 있는 일이다.

## 피드백

이 문서를 더 개선할 수 있도록 의견이나 질문을 남겨 주기 바란다. 교신 저자 Jacob Austin에게 jacobaustin123 [at] gmail [dot] com으로 연락하거나, [GitHub](https://github.com/jax-ml/scaling-book)에 issue, pull request, discussion을 올려 수정을 제안할 수 있다.
