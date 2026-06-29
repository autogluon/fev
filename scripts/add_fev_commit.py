# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas",
#     "GitPython",
# ]
# ///
"""Add (or overwrite) the ``fev_commit`` column in benchmark summary CSVs.

By default the commit is the latest commit on the ``main`` branch, but it can
also be provided manually via ``--commit``. The column is appended only to the
summary files passed on the command line, so it is safe to apply it to a subset.

Run with ``uv run`` so the dependencies in the inline metadata block above are
installed automatically:

    # Fill the latest commit on 'main' into two summaries
    uv run scripts/add_fev_commit.py benchmarks/fev_bench/results/toto-2.0.csv \\
        benchmarks/fev_bench/results/tabpfn-ts-3.csv

    # Use an explicit commit hash
    uv run scripts/add_fev_commit.py --commit 1970c03 path/to/summary.csv

    # Overwrite an existing fev_commit column (otherwise an error is raised)
    uv run scripts/add_fev_commit.py --overwrite path/to/summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import git
import pandas as pd

# Length of the abbreviated commit hash stored in the summaries (e.g. "f323f6c").
SHORT_HASH_LEN = 7

# Branch whose latest commit is recorded by default.
DEFAULT_BRANCH = "main"


def get_latest_commit(repo_path: Path, branch: str = DEFAULT_BRANCH) -> str:
    """Return the abbreviated hash of the latest commit on ``branch``.

    Resolves the branch explicitly (rather than the checked-out HEAD) so the
    recorded commit does not depend on which branch is currently checked out.
    """
    repo = git.Repo(repo_path, search_parent_directories=True)
    return repo.commit(branch).hexsha[:SHORT_HASH_LEN]


def resolve_commit(repo_path: Path, commit: str) -> str:
    """Validate and abbreviate a user-provided commit-ish."""
    repo = git.Repo(repo_path, search_parent_directories=True)
    return repo.commit(commit).hexsha[:SHORT_HASH_LEN]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", type=Path, help="Summary CSV files to update.")
    parser.add_argument(
        "--commit",
        default=None,
        help=f"Commit to record (any git ref). Defaults to the latest commit on the '{DEFAULT_BRANCH}' branch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the fev_commit column if it already has values (default: raise an error instead).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_path = Path(__file__).resolve().parent

    if args.commit is None:
        commit = get_latest_commit(repo_path)
        print(f"Using latest commit on '{DEFAULT_BRANCH}': {commit}")
    else:
        commit = resolve_commit(repo_path, args.commit)
        print(f"Using provided commit: {commit}")

    for path in args.summaries:
        if not path.exists():
            print(f"  skip (not found): {path}", file=sys.stderr)
            continue

        df = pd.read_csv(path)
        if "fev_commit" in df.columns and df["fev_commit"].notna().any() and not args.overwrite:
            raise SystemExit(f"Error: {path} already has a fev_commit column. Pass --overwrite to replace it.")

        df["fev_commit"] = commit
        df.to_csv(path, index=False)
        print(f"  updated ({len(df)} rows): {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
