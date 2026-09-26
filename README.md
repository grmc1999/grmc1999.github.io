# grmc1999.github.io

Personal Jekyll website.

## Local development

1. Install Ruby + Bundler.
2. Download closed-object sample point clouds (bunny/dragon/teapot):
   ```bash
   python3 scripts/download_example_point_clouds.py
   ```
3. Install dependencies:
   ```bash
   bundle install
   ```
4. Run link/path checks:
   ```bash
   python3 scripts/check_local_links.py
   ```
5. Build site:
   ```bash
   bundle exec jekyll build
   ```

## Deployment checks

GitHub Actions workflow at `.github/workflows/jekyll.yml` runs:
- local link/path validation (`scripts/check_local_links.py`)
- Jekyll build (`bundle exec jekyll build`)

## Adding a question

Add each research or intellectual question as a Markdown file in `_questions/`.
The body contains the discussion, while related articles, definitions, or books
are listed in the `sources` front matter:

```yaml
---
title: "Can learned representations separate causality from correlation?"
date: 2026-09-26
sources:
  - title: "The Book of Why"
    authors: "Judea Pearl and Dana Mackenzie"
    year: 2018
    type: Book
  - title: "A research article"
    authors: "Example Author"
    publication: "Example Journal"
    year: 2026
    type: Article
    url: "https://example.org/article"
---

The question body goes here.
```

Question pages automatically use the `question` layout, and the Questions
index lists published entries in reverse chronological order.
