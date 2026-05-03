"""
DAG layering via Kahn's algorithm on shared_state dependencies.

Files are partitioned into layers where files within a layer have no
inter-dependencies and can be generated in parallel (§4.3).
"""

from __future__ import annotations

from typing import Dict, List, Set

from matrioska.core.state import FileSpec


def compute_layers(files: List[FileSpec]) -> List[List[FileSpec]]:
    """Partition files into topological layers for parallel generation.

    A file depends on another if it reads a key the other writes.
    Files in the same layer have no mutual dependencies.

    Returns:
        List of layers (each layer is a list of FileSpecs).
        Files within a layer can be generated in parallel.
    """
    # Map writer key → FileSpec that writes it (first writer wins)
    writers: Dict[str, FileSpec] = {}
    for f in files:
        for k in f.shared_state_writes:
            if k not in writers:
                writers[k] = f

    # Build dependency graph: file_name → {dependent_file_names}
    deps: Dict[str, Set[str]] = {f.name: set() for f in files}
    for f in files:
        for k in f.shared_state_reads:
            w = writers.get(k)
            if w and w.name != f.name:
                deps[f.name].add(w.name)

    # Kahn's algorithm
    layers: List[List[FileSpec]] = []
    remaining: Dict[str, FileSpec] = {f.name: f for f in files}

    while remaining:
        ready = [
            f for f in remaining.values() if deps[f.name].isdisjoint(remaining.keys())
        ]

        if not ready:
            # Cycle or missing writer: break by picking lowest-order file
            ready = [min(remaining.values(), key=lambda x: x.order)]

        ready.sort(key=lambda x: x.order)
        layers.append(ready)

        for f in ready:
            del remaining[f.name]

    return layers


def validate_dag(files: List[FileSpec]) -> Dict[str, List[str]]:
    """Validate the DAG: check that all read keys are written by SOME file.

    Returns:
        Dict mapping file name to list of missing key issues.
    """
    all_writes: Set[str] = set()
    for f in files:
        all_writes.update(f.shared_state_writes)

    issues: Dict[str, List[str]] = {}
    for f in files:
        missing = [k for k in f.shared_state_reads if k not in all_writes]
        if missing:
            issues[f.filename] = missing

    return issues


def get_dependency_chain(files: List[FileSpec]) -> Dict[str, List[str]]:
    """Build a dependency chain dict: filename → [dependent filenames]."""
    writers: Dict[str, str] = {}
    for f in files:
        for k in f.shared_state_writes:
            if k not in writers:
                writers[k] = f.filename

    chain: Dict[str, List[str]] = {f.filename: [] for f in files}
    for f in files:
        for k in f.shared_state_reads:
            w = writers.get(k)
            if w and w != f.filename:
                chain[f.filename].append(w)

    return chain
