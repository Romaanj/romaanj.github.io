---
title: "Paper Visualization"
summary: "논문 하나를 완전히 이해할 때까지 뜯어보고 그리는 프로젝트. 같은 13개 질문으로 읽고 각 질문에 한 문장으로 답한 뒤, 문장으로 부족한 자리마다 인터랙티브 그림을 넣는다. 그림 속 숫자는 우리 A100에서 돌린 probe 실측값. 현재 DFlash(ICML 2026), GPTQ(ICLR 2023)."
status: active
date: 2026-08-20
tags: ["visualization", "paper reading", "inference", "quantization"]
link: "/projects/paper-viz/"
---

논문을 두 번 읽는 것과 이해하는 것은 다르다. 메커니즘을 직접 그릴 수 있을 때까지
뜯어보고, 그 과정을 그대로 페이지로 만든다.

읽기 뼈대는 [13개 질문](https://gisbi-kim.github.io/paper-reading-13-questions/)
(배경 → 문제 → 기존 한계 → 목표 → 방법 → 핵심 아이디어 → 검증 → 결과 → 비교 →
의의 → 한계 → 향후 과제 → 자원 공개). 각 질문에 볼드 한 문장으로 답하고, 그 사이사이에
개념 그림을 넣는다. GPU는 결과 재연이 아니라 그림에 넣을 실제 값(tensor, token, 밀리초)을
뽑는 probe로만 쓴다.

[페이지 보러 가기 →](/projects/paper-viz/)
