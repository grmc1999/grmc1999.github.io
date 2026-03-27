# Deployment review (March 26, 2026)

## Technical deployment issues found

1. **GitHub Pages base URL likely misconfigured**
   - `_config.yml` sets `baseurl: /home`.
   - For a user site named `grmc1999.github.io`, `baseurl` should typically be empty (`""`).
   - A non-empty base URL can break CSS and internal page links in production.

2. **Navigation links bypass `baseurl`**
   - `_includes/header.html` hardcodes internal links as `/` and `/summaries`.
   - If `baseurl` changes (or is required in a project site), these links can break.

3. **`site.url` is not set while canonical/feed URLs are generated from it**
   - `_includes/head.html` builds canonical and RSS URLs using `site.url` + `site.baseurl`.
   - Without `url` in `_config.yml`, generated canonical links can be relative or incomplete.

4. **Legacy Google Analytics snippet**
   - `_includes/head.html` uses `analytics.js` with a legacy UA property (`UA-75587219-1`).
   - Modern deployments should use Google tag (`gtag.js`) and GA4 measurement IDs.

5. **Broken relative asset paths in multiple nested markdown pages**
   - Several pages reference `assets/img/...` without a leading slash or `relative_url` filter.
   - On nested routes (e.g., `/learning/latent/`) these resolve to non-existent paths like `/learning/latent/assets/img/...`.

6. **No local build/dependency lock setup for reproducible deployment checks**
   - No `Gemfile`/`Gemfile.lock` found in repository root.
   - Running `jekyll build` in this environment fails because `jekyll` is not installed.

## Proposed improvements

1. **Fix URL config first**
   - In `_config.yml`, set:
     - `url: "https://grmc1999.github.io"`
     - `baseurl: ""` (for user site) or `"/<repo-name>"` (for project site)

2. **Make all internal links baseurl-safe**
   - Replace hardcoded links in templates with Liquid helpers, for example:
     - `{{ '/' | relative_url }}`
     - `{{ '/summaries/' | relative_url }}`

3. **Normalize image/link paths in markdown**
   - For all local assets, use one of:
     - absolute-root style: `/assets/img/example.png`
     - Liquid style: `{{ '/assets/img/example.png' | relative_url }}`

4. **Upgrade analytics integration**
   - Replace legacy UA snippet with GA4 (`gtag.js`) and store ID in `_config.yml` or `_data/options.yml`.

5. **Add reproducible local build tooling**
   - Add `Gemfile` pinned to `github-pages` (or explicit `jekyll` + plugins).
   - Add npm-free CI check (GitHub Actions) to run `bundle exec jekyll build` on pushes/PRs.

6. **Add automated broken-link/static-path validation**
   - Add a lightweight script or link checker to catch nested-route path issues before deploy.

## Commands used during this audit

- `rg --files`
- `jekyll build --trace` (failed: jekyll not installed in current environment)
- A Python one-off script to detect missing local references in markdown/html
