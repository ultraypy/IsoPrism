"""Prepare fixed source-only catalogues and held-out benchmark partitions."""
import argparse
import json
from pathlib import Path

from short2long import unified_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(unified_data.SPECS))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tasks = args.tasks or list(unified_data.SPECS)
    for task in tasks:
        spec = config["datasets"][task]
        unified_data.SPECS[task] = (str((args.data_root / spec["path"]).resolve()),
                                   spec["column"], spec["holdout"])
        unified_data.prepare(task, args.output_root.resolve())


if __name__ == "__main__":
    main()
