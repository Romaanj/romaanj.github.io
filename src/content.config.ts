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
  }),
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

export const collections = { reviews, study, projects, talks };
