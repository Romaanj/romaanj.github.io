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

| field     | type                       | notes                                   |
|-----------|----------------------------|-----------------------------------------|
| `title`   | string (required)          | paper title                             |
| `summary` | string (required)          | one-liner shown on cards                |
| `date`    | date (required)            | `YYYY-MM-DD`                            |
| `arxivId` | string?                    | e.g. `2404.12345`                       |
| `authors` | string?                    |                                          |
| `lab`     | string?                    |                                          |
| `venue`   | string?                    |                                          |
| `tags`    | string[] (default `[]`)    |                                          |
| `links`   | string[] (default `[]`)    | slugs of related reviews → graph edges  |
| `source`  | `autosweep` \| `manual`    | default `manual`                        |
| `rating`  | number? (1–5)              |                                          |

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
summary: "One-line takeaway."
links: []
source: manual
---

## What it does
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

It writes `src/content/reviews/<slug>.md` with `source: autosweep`, a `TODO` summary, and an
empty *What it does / How it works / Why it matters / Open questions* body template. It
refuses (exit 1) if the slug already exists. The daily lit autosweep calls this for each
public-worthy paper, then the review body is written in a public-facing tone, and the result
is committed to `src/content/reviews/` and pushed — the Pages action publishes it.
