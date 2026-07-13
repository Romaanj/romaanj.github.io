# romaanj.github.io

Personal research-lab site of **Junwon Lee** (IRIS Lab, SKKU) — training-free acceleration of
diffusion LLMs, KV-cache compression & quantization, efficient inference systems.

Built with [Astro 5](https://astro.build) (static output). Live at
**https://romaanj.github.io**.

## Local development

```bash
npm install
npm run dev        # dev server (astro dev --host), http://localhost:4321
npm run build      # static build into dist/
npm run preview    # serve the built dist/ locally
```

## Layout

```
src/
  layouts/Base.astro        # page shell (header, footer, fonts, meta)
  components/               # Header / Footer
  styles/global.css         # design system (do not fork per-page styles into it)
  pages/                    # routes; interactive study pages are plain .astro files here
  content/                  # markdown collections (see below)
    reviews/  study/  projects/
  content.config.ts         # collection schemas (zod)
scripts/
  publish_review.py         # sanitized review-stub generator (see "Review pipeline")
```

## Content model

Three collections, defined in `src/content.config.ts`. A file's **slug is its filename**
without extension (`src/content/reviews/my-paper.md` → `/reviews/my-paper/`).

### `reviews` — paper reviews

| field        | type                              | notes                                                        |
|--------------|-----------------------------------|--------------------------------------------------------------|
| `title`      | string (required)                 | paper title                                                  |
| `summary`    | string (required)                 | one-liner shown on cards (English)                           |
| `summary_ko` | string?                           | Korean one-liner (shown when the language toggle is KO)      |
| `date`       | date (required)                   | `YYYY-MM-DD`                                                 |
| `arxivId`    | string?                           | e.g. `2404.12345`                                            |
| `authors`    | string?                           |                                                               |
| `lab`        | string?                           |                                                               |
| `venue`      | string?                           |                                                               |
| `tags`       | string[] (default `[]`)           |                                                               |
| `topic`      | string?                           | reading-list section: `diffusion-llm` \| `kv-cache` \| `hybrid-architecture` \| `post-training` \| `on-device` \| `architecture` \| `compression` \| `serving` |
| `links`      | string[] (default `[]`)           | slugs of related reviews → graph edges                       |
| `resources`  | `{label, url}[]` (default `[]`)   | **verified** primary links only (arXiv / PDF / project / GitHub / dataset / checkpoint) |
| `figures`    | `{src, caption, caption_ko?, credit}[]` (default `[]`) | 1–2 key figures pulled from the paper's arXiv HTML, served from `public/figures/<slug>/`; always credited (`Figure N from arXiv:<id> — authors' figure`), ≤ 2 MB each |
| `analysis`   | `{ ko: {...}, en: {...} }`?       | bilingual 13-key structured analysis (below)                 |
| `thread`     | `{ko, en}`?                       | narrative research-thread section: `\|-` block scalars, 3–4 blank-line-separated paragraphs (lineage → conceptual shift → what it opens), same content in both languages |
| `sparks`     | `{ko, en}[]` (default `[]`)       | 2–3 bilingual idea seeds, strictly grounded in the paper's stated limitations/future work |
| `source`     | `autosweep` \| `manual`           | default `manual`                                             |
| `rating`     | number? (1–5)                     |                                                               |

**The 13-item analysis block (enriched format).** `analysis.ko` and
`analysis.en` each carry the same 13 keys in a **fixed order**, **2–4
sentences per key** — the first sentence is the crisp information-dense claim,
the rest add mechanism/context, with an apt analogy on a handful of items
where it genuinely clarifies (identical content across the two languages).
Key quantitative numbers may be marked `**bold**`; the renderer converts them.

| # | key            | ko label    | en label            |
|---|----------------|-------------|----------------------|
| 1 | `background`   | 배경        | Background           |
| 2 | `problem`      | 문제        | Problem              |
| 3 | `prior_limits` | 기존 한계   | Prior limitations    |
| 4 | `goal`         | 목표        | Goal                 |
| 5 | `method`       | 방법        | Method               |
| 6 | `key_idea`     | 핵심 아이디어 | Key idea           |
| 7 | `validation`   | 검증        | Validation           |
| 8 | `results`      | 결과        | Results              |
| 9 | `comparison`   | 비교        | Comparison           |
| 10 | `significance` | 의의       | Significance         |
| 11 | `limitations`  | 한계       | Limitations          |
| 12 | `future_work`  | 향후 과제  | Future work          |
| 13 | `resources`    | 자원 공개  | Released resources   |

**Language toggle.** The review page renders one language at a time — English
by default, Korean via the KO/EN toggle. The toggle switches the 13 labels and
sentence bodies together (and `summary` ↔ `summary_ko` where shown); both maps
must therefore be complete.

**Analysis YAML shape** (mini-example, first keys shown — all 13 required in
each language):

```yaml
analysis:
  ko:
    background: 'LLM 디코딩은 KV 캐시 메모리에 의해 병목이 걸린다.'
    problem: '긴 컨텍스트에서 KV 캐시가 배치 크기를 제한한다.'
    prior_limits: '기존 4-bit 양자화는 **2-bit**에서 정확도가 붕괴한다.'
    # ... goal, method, key_idea, validation, results, comparison,
    #     significance, limitations, future_work, resources
  en:
    background: 'LLM decoding is bottlenecked by KV-cache memory.'
    problem: 'At long context the KV cache caps batch size.'
    prior_limits: 'Prior 4-bit quantizers collapse at **2-bit**.'
    # ... same 13 keys, same order, same content as ko
```

### `study` — study notes

| field         | type                                   | notes                                             |
|---------------|----------------------------------------|---------------------------------------------------|
| `title`       | string (required)                      |                                                    |
| `track`       | `models` \| `quantization` \| `gpu`    | which study track the note belongs to             |
| `summary`     | string (required)                      |                                                    |
| `date`        | date (required)                        |                                                    |
| `order`       | number (default 99)                    | sort position within the track                    |
| `interactive` | string?                                | route of a custom interactive page (see below)    |
| `tags`        | string[] (default `[]`)                |                                                    |

### `projects` — side projects

| field     | type                                        | notes            |
|-----------|----------------------------------------------|------------------|
| `title`   | string (required)                            |                  |
| `summary` | string (required)                            |                  |
| `status`  | `idea` \| `active` \| `paused` \| `done`     | default `idea`   |
| `date`    | date (required)                              |                  |
| `tags`    | string[] (default `[]`)                      |                  |
| `repo`    | string?                                      | GitHub URL       |

## Adding content

**Review** — create `src/content/reviews/<slug>.md`:

```markdown
---
title: "Paper Title"
arxivId: "2404.12345"
date: 2026-07-04
tags: [kv-cache, quantization]
topic: kv-cache
summary: "One-line takeaway."
summary_ko: "한 줄 요약."
links: []
resources:
  - { label: "arXiv", url: "https://arxiv.org/abs/2404.12345" }
analysis:
  ko: { background: '...', ... }   # all 13 keys, fixed order (see table above)
  en: { background: '...', ... }
source: manual
---

## Notes
...
```

**Study note** — create `src/content/study/<slug>.md` with `track`, `order`, etc.
Regular notes render as markdown articles.

**Interactive study pages** are *not* markdown: they are plain `.astro` pages under
`src/pages/study/` (custom HTML/CSS/JS visualizations). To surface one in the study index,
add a stub entry to the `study` collection with `interactive` set to the page's route
(e.g. `interactive: "/study/gpu-memory-wall/"`); the index links there instead of rendering
a markdown body.

**Project** — create `src/content/projects/<slug>.md`.

## Deploy

Push to `main` → GitHub Actions builds (`withastro/action`) → deploys to GitHub Pages
(`actions/deploy-pages`). Workflow: `.github/workflows/deploy.yml` (also runnable manually
via *Actions → Deploy to GitHub Pages → Run workflow*).

**One-time setup**

1. Create the repo `Romaanj/romaanj.github.io` and push this directory as its root
   (branch `main`).
2. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Done — every subsequent push to `main` publishes automatically.

## Review pipeline (auto-publish)

`scripts/publish_review.py` turns an internal lit note into a **sanitized public stub** —
it copies only the paper title and arXiv id, never the note's prose:

```bash
python3 scripts/publish_review.py <lit-note.md> --slug <slug> [--date YYYY-MM-DD] [--tags a,b]
```

It writes `src/content/reviews/<slug>.md` with `source: autosweep` and the full new-schema
frontmatter as TODO stubs: `topic`, `summary` / `summary_ko`, `links: []`, `resources: []`
(verified links only), `figures: []` (1–2 credited key figures from the paper's arXiv HTML,
saved to `public/figures/<slug>/`), a complete bilingual `analysis:` block (`ko:` +
`en:`, each with all 13 keys in fixed order, every value `'TODO'`), bilingual `thread:`
block-scalar TODOs, and `sparks: []`, plus a minimal `## Notes` body. It refuses (exit 1) if the slug already exists. The daily lit autosweep
calls this for each public-worthy paper, then the 13-item analysis is filled in both
languages from the paper itself in the enriched format (2–4 sentences per key, first
sentence = the crisp claim) in a public-facing tone, figures are downloaded and credited,
and the result is committed to `src/content/reviews/` (+ `public/figures/`) and pushed —
the Pages action publishes it.
