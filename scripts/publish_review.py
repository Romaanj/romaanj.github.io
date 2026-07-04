#!/usr/bin/env python3
"""publish_review.py — create a SANITIZED public review stub from an internal lit note.

Usage:
    python3 scripts/publish_review.py <lit-note.md> --slug <slug> \
        [--date YYYY-MM-DD] [--tags a,b]

What it does:
  * Reads the internal note ONLY to extract two public facts:
      - the paper title (frontmatter `title:` or first `# H1`, else derived
        from the filename)
      - the arXiv id (explicit `arXiv:`/`arxiv.org` reference in the body,
        or a bare id in the filename/body)
  * Emits src/content/reviews/<slug>.md with the full reviews frontmatter
    (source: autosweep, summary/summary_ko placeholders, topic/links/resources
    stubs, and a bilingual 13-item analysis block with every value TODO) plus a
    minimal body template. It deliberately NEVER copies note prose — internal
    notes contain private research commentary that must not be published.
  * Refuses (exit 1) if the output slug already exists.
  * Prints the created path on success.

Stdlib only. No third-party dependencies.
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

HOMEPAGE_ROOT = Path(__file__).resolve().parent.parent
REVIEWS_DIR = HOMEPAGE_ROOT / "src" / "content" / "reviews"

# arXiv id core: YYMM.NNNN(N) with a plausible month, optional version suffix.
ARXIV_CORE = r"\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}"
ARXIV_EXPLICIT_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv\s*:\s*)(" + ARXIV_CORE + r")(?:v\d+)?",
    re.IGNORECASE,
)
ARXIV_BARE_RE = re.compile(r"(?<![\d.])(" + ARXIV_CORE + r")(?:v\d+)?(?![\d.])")

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Allowed values for the `topic` frontmatter field (reading-list sections).
TOPIC_VALUES = (
    "diffusion-llm | kv-cache | hybrid-architecture | post-training | "
    "on-device | architecture | compression | serving"
)

# The 13 analysis keys, FIXED ORDER — must match src/content.config.ts.
ANALYSIS_KEYS = (
    "background",
    "problem",
    "prior_limits",
    "goal",
    "method",
    "key_idea",
    "validation",
    "results",
    "comparison",
    "significance",
    "limitations",
    "future_work",
    "resources",
)

BODY_TEMPLATE = """
## Notes

<!-- TODO: optional free-form notes; the structured 13-item analysis lives in the frontmatter. -->
"""


def fail(msg: str) -> "NoReturn":  # noqa: F821 - py3.8-friendly annotation
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def strip_markdown(s: str) -> str:
    """Flatten light markdown in a title line to plain text."""
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # [text](url) -> text
    s = s.replace("**", "").replace("__", "").replace("`", "")
    return " ".join(s.split()).strip()


def parse_title(text: str, fallback: str) -> str:
    # 1) frontmatter `title:` if the note starts with a --- block
    if text.lstrip().startswith("---"):
        m = re.match(r"\s*---\s*\n(.*?)\n\s*---\s*(?:\n|$)", text, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                km = re.match(r"^title\s*:\s*(.+?)\s*$", line)
                if km:
                    val = km.group(1).strip().strip("\"'").strip()
                    if val:
                        return strip_markdown(val)
    # 2) first H1 heading
    for line in text.splitlines():
        hm = re.match(r"^#\s+(.+?)\s*$", line)
        if hm:
            return strip_markdown(hm.group(1))
    # 3) fallback (derived from filename)
    return fallback


def find_arxiv_id(note_path: Path, text: str) -> str:
    # explicit reference in the body wins
    m = ARXIV_EXPLICIT_RE.search(text)
    if m:
        return m.group(1)
    # bare id in the filename
    m = ARXIV_BARE_RE.search(note_path.name)
    if m:
        return m.group(1)
    # bare id anywhere in the body (last resort)
    m = ARXIV_BARE_RE.search(text)
    if m:
        return m.group(1)
    return ""


def yaml_str(s: str) -> str:
    """Safe double-quoted YAML scalar (JSON strings are valid YAML flow scalars)."""
    return json.dumps(s, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert an internal lit note into a sanitized public review stub."
    )
    ap.add_argument("note", help="path to the internal lit note (.md)")
    ap.add_argument("--slug", required=True, help="output slug (kebab-case)")
    ap.add_argument("--date", default=None, help="review date, YYYY-MM-DD (default: today)")
    ap.add_argument("--tags", default="", help="comma-separated tags, e.g. kv-cache,dllm")
    args = ap.parse_args()

    note_path = Path(args.note).expanduser()
    if not note_path.is_file():
        fail(f"lit note not found: {note_path}")

    slug = args.slug.strip()
    if not SLUG_RE.fullmatch(slug):
        fail(f"invalid slug {slug!r}: use lowercase kebab-case, e.g. my-paper-2404")

    if args.date is not None:
        try:
            date = datetime.date.fromisoformat(args.date)
        except ValueError:
            fail(f"invalid --date {args.date!r}: expected YYYY-MM-DD")
    else:
        date = datetime.date.today()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    for ext in (".md", ".mdx"):
        existing = REVIEWS_DIR / f"{slug}{ext}"
        if existing.exists():
            fail(
                f"refusing to overwrite: review slug '{slug}' already exists at "
                f"{existing}. Pick a different --slug or edit the existing review."
            )

    try:
        text = note_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        fail(f"could not read {note_path}: {e}")

    fallback_title = note_path.stem.replace("-", " ").replace("_", " ").strip() or slug
    title = parse_title(text, fallback_title)
    arxiv_id = find_arxiv_id(note_path, text)

    lines = ["---", f"title: {yaml_str(title)}"]
    if arxiv_id:
        lines.append(f"arxivId: {yaml_str(arxiv_id)}")
    lines.append(f"date: {date.isoformat()}")
    lines.append("tags: [" + ", ".join(yaml_str(t) for t in tags) + "]")
    lines.append(f"topic: ''  # TODO: one of {TOPIC_VALUES}")
    lines.append('summary: "TODO one-line summary (English)"')
    lines.append("summary_ko: 'TODO'")
    lines.append("links: []  # slugs of related reviews already on the site")
    lines.append("resources: []  # verified primary links only ({label, url}); curl -sI 200 before adding")
    lines.append("analysis:")
    for lang in ("ko", "en"):
        lines.append(f"  {lang}:")
        for key in ANALYSIS_KEYS:
            lines.append(f"    {key}: 'TODO'")
    lines.append('source: "autosweep"')
    lines.append("---")
    content = "\n".join(lines) + "\n" + BODY_TEMPLATE

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REVIEWS_DIR / f"{slug}.md"
    out_path.write_text(content, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
