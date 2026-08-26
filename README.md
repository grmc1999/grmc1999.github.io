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
