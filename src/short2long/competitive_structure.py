"""Source-catalogue terminal descriptors without expression or held-out labels."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .model import splice_structure_tensors
from .structure import _attribute, _open_text


ANNOTATIONS = {
    "cross_donor": "data/raw/crc_cross_donor/extracted/gffcmp.query.all_samples.combined.colored.sorted.gff3",
    "cross_disease": "data/raw/crc_cross_donor/extracted/gffcmp.query.all_samples.combined.colored.sorted.gff3",
    "spatial": "data/raw/post_mi_spatial/gffcmp.multi_exons.annotated.gtf",
}
FEATURE_NAMES = ["log_five_prime_terminal_exon_length", "log_three_prime_terminal_exon_length",
                 "log_transcript_span", "log_exonic_length", "terminal_annotation_available"]


@lru_cache(maxsize=6)
def annotation_terminals(path, aliases):
    """Read public transcript/exon annotations for the requested identifiers only."""
    keep = set(aliases)
    records, exons = {}, defaultdict(list)
    with _open_text(Path(path)) as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] not in {"transcript", "mRNA", "exon"}:
                continue
            attrs = fields[8]
            if fields[2] in {"transcript", "mRNA"}:
                name = _attribute(attrs, "ID") or _attribute(attrs, "transcript_id")
                name = name.removeprefix("transcript:") if name else None
                if name in keep:
                    record = (fields[0], fields[6], int(fields[3]), int(fields[4]))
                    if name in records and records[name] != record:
                        raise ValueError(f"Conflicting transcript coordinates for {name}")
                    records[name] = record
            else:
                parent = _attribute(attrs, "Parent") or _attribute(attrs, "transcript_id")
                for name in (parent or "").split(","):
                    name = name.removeprefix("transcript:")
                    if name in keep:
                        exons[name].append((int(fields[3]), int(fields[4])))
    output = {}
    for name, (chrom, strand, start, end) in records.items():
        ordered = sorted(set(exons.get(name, [])))
        if not ordered or strand not in {"+", "-"}:
            continue
        if any(a > b or a < start or b > end for a, b in ordered):
            raise ValueError(f"Invalid exon bounds for {name}")
        if any(ordered[i][1] >= ordered[i + 1][0] for i in range(len(ordered) - 1)):
            raise ValueError(f"Overlapping exons for {name}")
        left, right = ordered[0][1] - ordered[0][0] + 1, ordered[-1][1] - ordered[-1][0] + 1
        five, three = (left, right) if strand == "+" else (right, left)
        output[name] = {"chromosome": chrom, "strand": strand, "start": start, "end": end,
                        "features": np.log1p([five, three, end - start + 1,
                                             sum(b - a + 1 for a, b in ordered)]).tolist()}
    return output


def terminal_features(task, view, project_root):
    """Standardize within the retained SOURCE catalogue, separately for inner/final.

    Harmonized intron-chain targets merge endpoints; they deliberately receive an
    unavailable mask, not endpoints borrowed from an arbitrarily chosen transcript.
    """
    values = np.zeros((len(view["isoforms"]), 5), np.float32)
    annotation = ANNOTATIONS.get(task)
    metadata = {"feature_names": FEATURE_NAMES, "annotation": annotation, "known": 0,
                "catalogue_size": len(values), "normalization_population": "retained source catalogue",
                "merged_chain_endpoints": "unavailable; never imputed from target expression"}
    if annotation is not None:
        aliases = tuple(str(name).rsplit(":", 1)[0] if task == "spatial" else str(name)
                        for name in view["isoforms"])
        records = annotation_terminals(str(Path(project_root) / annotation), aliases)
        for i, alias in enumerate(aliases):
            record = records.get(alias)
            if record:
                values[i, :4] = record["features"]
                values[i, 4] = 1
    valid = values[:, 4] == 1
    metadata["known"] = int(valid.sum())
    if valid.any():
        mean, sd = values[valid, :4].mean(0), values[valid, :4].std(0)
        sd[sd < 1e-5] = 1
        values[valid, :4] = (values[valid, :4] - mean) / sd
        metadata.update(mean=mean.tolist(), sd=sd.tolist())
    metadata["chain_collisions"] = representation_collisions(view["groups"], view["structures"])
    metadata["remaining_collisions"] = representation_collisions(view["groups"], view["structures"], values)
    return values, metadata


def representation_collisions(groups, structures, additional=None):
    ids, offsets, descriptors = splice_structure_tensors(structures)
    bins = defaultdict(list)
    for i, gene in enumerate(groups):
        end = int(offsets[i + 1]) if i + 1 < len(offsets) else len(ids)
        key = (int(gene), tuple(sorted(ids[int(offsets[i]):end].tolist())), tuple(descriptors[i].tolist()))
        if additional is not None:
            key += (tuple(np.asarray(additional[i]).tolist()),)
        bins[key].append(i)
    return [indices for indices in bins.values() if len(indices) > 1]
