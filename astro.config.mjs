import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://romaanj.github.io',
  integrations: [
    mdx(),
    // /research/ is hidden from nav for now — keep it out of the sitemap too
    sitemap({ filter: (page) => !page.includes('/research/') }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      [
        rehypeKatex,
        {
          // scaling-book chapters \def color macros in their first math block;
          // rehype-katex renders each block independently, so define them globally.
          macros: {
            '\\red': '\\textcolor{red}{#1}',
            '\\green': '\\textcolor{green}{#1}',
            '\\blue': '\\textcolor{blue}{#1}',
            '\\purple': '\\textcolor{purple}{#1}',
            '\\orange': '\\textcolor{orange}{#1}',
            '\\gray': '\\textcolor{gray}{#1}',
          },
        },
      ],
    ],
    shikiConfig: {
      theme: 'github-light',
    },
  },
});
