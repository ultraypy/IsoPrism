from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import TextIO

import numpy as np


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _attribute(attributes: str, key: str) -> str | None:
    gff = re.search(rf"(?:^|;\s*){re.escape(key)}=([^;]+)", attributes)
    if gff:
        return gff.group(1).strip().strip('"')
    gtf = re.search(rf'(?:^|;\s*){re.escape(key)}\s+"([^"]+)"', attributes)
    return gtf.group(1) if gtf else None


def structures_from_annotation(
    path: Path,
    isoform_ids: np.ndarray,
    isoform_genes: np.ndarray,
    strip_colon_suffix: bool = False,
) -> tuple[np.ndarray, int]:
    """Recover deterministic full intron-chain descriptors from GFF3/GTF.

    Multi-exon transcripts are represented by their complete ordered intron
    chain. A single-exon transcript uses its transcript span as a structural
    interval so it still receives a reproducible non-ID structural encoding.
    """
    aliases = [
        str(value).rsplit(":", 1)[0] if strip_colon_suffix else str(value)
        for value in isoform_ids
    ]
    keep = set(aliases)
    transcripts: dict[str, tuple[str, str, int, int]] = {}
    exons: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            feature, attributes = fields[2], fields[8]
            if feature == "transcript":
                transcript = _attribute(attributes, "ID") or _attribute(
                    attributes, "transcript_id"
                )
                if transcript:
                    transcript = transcript.removeprefix("transcript:")
                if transcript in keep:
                    transcripts[transcript] = (
                        fields[0],
                        fields[6],
                        int(fields[3]),
                        int(fields[4]),
                    )
            elif feature == "exon":
                parent = _attribute(attributes, "Parent") or _attribute(
                    attributes, "transcript_id"
                )
                if parent:
                    parent = parent.split(",", 1)[0].removeprefix("transcript:")
                if parent in keep:
                    exons[parent].append((int(fields[3]), int(fields[4])))

    structures: list[str] = []
    recovered = 0
    for original, alias, gene in zip(isoform_ids, aliases, isoform_genes):
        record = transcripts.get(alias)
        if record is None:
            structures.append(str(original))
            continue
        chromosome, strand, transcript_start, transcript_end = record
        ordered = sorted(exons.get(alias, []))
        if len(ordered) >= 2:
            intervals = [
                (ordered[index][1], ordered[index + 1][0])
                for index in range(len(ordered) - 1)
            ]
        else:
            intervals = [(transcript_start, transcript_end)]
        chain = ";".join(f"{start}-{end}" for start, end in intervals)
        structures.append(f"{gene}|{chromosome}|{strand}|{chain}")
        recovered += 1
    return np.asarray(structures, dtype=str), recovered
