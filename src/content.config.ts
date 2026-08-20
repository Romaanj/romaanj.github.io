import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

/**
 * The 13-item structured analysis (fixed order, one sentence each).
 * Keys map to: 배경/문제/기존 한계/목표/방법/핵심 아이디어/검증/결과/비교/의의/한계/향후 과제/자원 공개.
 */
const analysisBlock = z.object({
  background: z.string(),
  problem: z.string(),
  prior_limits: z.string(),
  goal: z.string(),
  method: z.string(),
  key_idea: z.string(),
  validation: z.string(),
  results: z.string(),
  comparison: z.string(),
  significance: z.string(),
  limitations: z.string(),
  future_work: z.string(),
  resources: z.string(),
});

/**
 * Build gate: reject an unfilled `publish_review.py` stub.
 *
 * The stub emits every frontmatter value as the literal 'TODO' (and `topic: ''`), all of
 * which are valid strings — so without this check a stub that lands in src/content/reviews/
 * builds green and publishes a page of placeholders. This walks the parsed frontmatter and
 * fails the build on any leftover placeholder. Frontmatter only: the body template's
 * `<!-- TODO: optional free-form notes -->` comment is invisible to the schema and stays legal.
 */
const PLACEHOLDER = /^\s*TODO\b/i;

const rejectUnfilledStub = (data: Record<string, unknown>, ctx: z.RefinementCtx) => {
  const walk = (value: unknown, path: (string | number)[]) => {
    if (typeof value === 'string') {
      if (PLACEHOLDER.test(value)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path,
          message: `unfilled stub placeholder ${JSON.stringify(value.trim().slice(0, 40))} — fill it in or delete the stub before publishing`,
        });
      }
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item, i) => walk(item, [...path, i]));
      return;
    }
    if (value && typeof value === 'object' && !(value instanceof Date)) {
      for (const [k, v] of Object.entries(value)) walk(v, [...path, k]);
    }
  };
  walk(data, []);

  if (typeof data.topic === 'string' && data.topic.trim() === '') {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['topic'],
      message:
        'empty topic — set one of diffusion-llm | kv-cache | hybrid-architecture | ' +
        'post-training | on-device | architecture | compression | serving',
    });
  }
};

/** Paper reviews — auto-published by the daily sweep or written by hand. */
const reviews = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/reviews' }),
  schema: z.object({
    title: z.string(),
    arxivId: z.string().optional(),
    authors: z.string().optional(),
    lab: z.string().optional(),
    venue: z.string().optional(),
    date: z.coerce.date(),
    tags: z.array(z.string()).default([]),
    /** reading-list section: diffusion-llm | kv-cache | hybrid-architecture |
     *  post-training | on-device | architecture | compression | serving */
    topic: z.string().optional(),
    summary: z.string(),
    summary_ko: z.string().optional(),
    /** slugs of related reviews — edges in the graph view */
    links: z.array(z.string()).default([]),
    /** verified primary sources only (paper / PDF / project / GitHub / dataset / checkpoint) */
    resources: z.array(z.object({ label: z.string(), url: z.string().url() })).default([]),
    /** key figures pulled from the paper (arXiv HTML), served from /public/figures/<slug>/ —
     *  always credited; captions bilingual */
    figures: z
      .array(
        z.object({
          src: z.string(),
          caption: z.string(),
          caption_ko: z.string().optional(),
          credit: z.string(),
        })
      )
      .default([]),
    /** bilingual 13-item structured analysis; en shown by default, ko via toggle */
    analysis: z.object({ ko: analysisBlock, en: analysisBlock }).optional(),
    /** narrative research-thread section: lineage → what this paper changes → what it opens.
     *  Multi-paragraph (blank-line separated), analogical, AGI-Papers-style. */
    thread: z.object({ ko: z.string(), en: z.string() }).optional(),
    /** 2-3 idea seeds STRICTLY grounded in the paper's stated limitations/future work —
     *  field-generic, never the owner's private research angles */
    sparks: z.array(z.object({ ko: z.string(), en: z.string() })).default([]),
    source: z.enum(['autosweep', 'manual']).default('manual'),
    rating: z.number().min(1).max(5).optional(),
  }).superRefine(rejectUnfilledStub),
});

/** Study notes — deep dives into models, quantization, GPU internals. */
const study = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/study' }),
  schema: z.object({
    title: z.string(),
    title_ko: z.string().optional(),
    track: z.enum(['models', 'quantization', 'gpu']),
    summary: z.string(),
    summary_ko: z.string().optional(),
    date: z.coerce.date(),
    order: z.number().default(99),
    /** set when the note lives as a custom interactive page instead of markdown */
    interactive: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
});

/** Side projects. */
const projects = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    status: z.enum(['idea', 'active', 'paused', 'done']).default('idea'),
    date: z.coerce.date(),
    tags: z.array(z.string()).default([]),
    repo: z.string().optional(),
    /** internal page the card links to (e.g. "/scaling-book/") */
    link: z.string().optional(),
  }),
});

/** Subtitled talks — English AI talks/keynotes with Korean subtitles.
 *  Videos are embedded from the original YouTube channel (never re-hosted);
 *  only the subtitle overlay is ours. Cues live in src/data/talks/<id>.cues.json. */
const talks = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/talks' }),
  schema: z.object({
    title: z.string(),
    title_ko: z.string().optional(),
    speaker: z.string(),
    affiliation: z.string().optional(),
    event: z.string().optional(),
    /** YouTube channel credit, e.g. "Kimi AI" */
    channel: z.string(),
    /** YouTube video id — embed only */
    videoId: z.string(),
    duration: z.string().optional(),
    date: z.coerce.date(),
    summary: z.string(),
    summary_ko: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
});

/** Korean translation of "How to Scale Your Model" (jax-ml/scaling-book, MIT).
 *  Full-text translation, chapter per entry. Figures are hot-linked from the
 *  original site; every page must link back to its original chapter URL and
 *  keep the MIT attribution block. */
const scalingBook = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/scaling-book' }),
  schema: z.object({
    /** chapter number as in the original book (0 = intro/index) */
    part: z.number(),
    title: z.string(),
    /** original English chapter title, shown as subtitle */
    title_en: z.string(),
    /** original chapter URL on jax-ml.github.io */
    original: z.string().url(),
    summary: z.string(),
    date: z.coerce.date(),
    /** translation status — drafted chapters only appear when true */
    published: z.boolean().default(true),
  }),
});

export const collections = { reviews, study, projects, talks, scalingBook };
