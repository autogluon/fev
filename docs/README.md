# Documentation

The docs are built with [MkDocs](https://www.mkdocs.org/) (Material theme) and versioned with
[mike](https://github.com/jimporter/mike). Config lives in `mkdocs.yml`; doc dependencies are in
`docs/requirements.txt`.

CI (`.github/workflows/docs.yml`) deploys automatically on every `v*` tag push. The commands below
are for building locally or deploying manually (e.g. re-deploying a release that's already on PyPI).

All commands use `uv` to create an ephemeral environment — the `--with-requirements` deps are layered
on top of the project itself (needed so `mkdocstrings` can import `fev`). Run them from the repo root.

## Preview locally

```bash
uv run --with-requirements docs/requirements.txt mkdocs serve
```

Opens a live-reloading preview at http://127.0.0.1:8000.

## Build only (no deploy)

```bash
uv run --with-requirements docs/requirements.txt mkdocs build
```

Outputs the static site to `site/`.

## Deploy manually

`mike` pushes the built site to the `gh-pages` branch. Replace `X.Y.Z` with the released version
(matching the `vX.Y.Z` tag), which also updates the `latest` alias:

```bash
uv run --with-requirements docs/requirements.txt mike deploy --push --update-aliases X.Y.Z latest
```
