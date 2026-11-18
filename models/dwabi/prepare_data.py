"""Download and verify the public FEV data revision used in the D-wabi run."""

import json
from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from shutil import copyfile

from huggingface_hub import hf_hub_download

MANIFEST = json.loads(Path(__file__).with_name("dataset_files.json").read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("mirror", type=Path)
    args = parser.parse_args()
    mirror = args.mirror.resolve()
    for config, entries in MANIFEST["files"].items():
        directory = mirror / config
        directory.mkdir(exist_ok=True, parents=True)
        for upstream, digest in entries.items():
            destination = directory / Path(upstream).name
            if not destination.exists():
                source = Path(hf_hub_download(
                    MANIFEST["repo"], upstream, revision=MANIFEST["revision"], repo_type="dataset",
                ))
                if sha256(source) != digest:
                    raise ValueError(f"{upstream}: the public archive no longer matches the recorded revision.")
                partial = destination.with_suffix(".part")
                copyfile(source, partial)
                partial.replace(destination)
            if sha256(destination) != digest:
                raise ValueError(f"{destination}: the local dataset does not match its recorded hash.")
    print(mirror)


if __name__ == "__main__":
    main()
