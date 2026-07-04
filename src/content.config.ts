import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

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
    summary: z.string(),
    /** slugs of related reviews — edges in the graph view */
    links: z.array(z.string()).default([]),
    source: z.enum(['autosweep', 'manual']).default('manual'),
    rating: z.number().min(1).max(5).optional(),
  }),
});

/** Study notes — deep dives into models, quantization, GPU internals. */
const study = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/study' }),
  schema: z.object({
    title: z.string(),
    track: z.enum(['models', 'quantization', 'gpu']),
    summary: z.string(),
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

export const collections = { reviews, study, projects };
