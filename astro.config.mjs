import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://romaanj.github.io',
  integrations: [
    mdx(),
    // /research/ is hidden from nav for now — keep it out of the sitemap too
    sitemap({ filter: (page) => !page.includes('/research/') }),
  ],
  markdown: {
    shikiConfig: {
      theme: 'github-light',
    },
  },
});
