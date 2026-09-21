"""Fetch official comparator repositories at the experiment's pinned revisions."""
import argparse
from pathlib import Path
import subprocess

from short2long.published_models import ROOT, UPSTREAM

URLS = {
    "rtdl": "https://github.com/yandex-research/rtdl-revisiting-models.git",
    "babel": "https://github.com/wukevin/babel.git",
    "scButterfly": "https://github.com/BioX-NKU/scButterfly.git",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for name, revision in UPSTREAM.items():
        path = Path(ROOT) / "third_party" / name
        if path.exists():
            actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
            if actual != revision or dirty:
                raise RuntimeError(f"Preserving existing {path}: wrong revision or local changes")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", URLS[name], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
        print(f"{name}: {revision}")


if __name__ == "__main__":
    main()
