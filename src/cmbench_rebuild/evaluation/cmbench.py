"""Strict, case-macro CMBench DINO aggregation contracts.

This module intentionally contains no model code.  It validates evaluator rows
before aggregation so a plausible-looking legacy CSV cannot silently become a
qualified golden result.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..dataset import task_id, normalized_box


def normalize_memory_level(value: object) -> str:
    level = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip()).strip("_")
    return level or "unspecified"


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def _annotation_rows(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = task_id(row)
        if not case_id:
            raise ValueError(f"{path}:{line_number}: annotation has no case_id")
        if case_id in indexed:
            duplicates.add(case_id)
        indexed[case_id] = row
    if duplicates:
        raise ValueError(f"duplicate annotation case_ids: {sorted(duplicates)}")
    return indexed


def eligible_case_ids(annotations_path: Path, expected_case_ids: Sequence[str]) -> tuple[str, ...]:
    """Return cases with a valid reference box under the benchmark denominator."""
    expected = tuple(str(value) for value in expected_case_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_case_ids must be non-empty and unique")
    annotations = _annotation_rows(annotations_path)
    missing = sorted(set(expected) - set(annotations))
    if missing:
        raise ValueError(f"annotation missing={missing}")
    eligible: list[str] = []
    for case_id in expected:
        row = annotations[case_id]
        status = str(row.get("status") or "ok").lower()
        if status in {"skip", "discard"} or bool(row.get("discard")):
            continue
        references = row.get("references")
        if isinstance(references, list) and references:
            for reference in references:
                if isinstance(reference, Mapping) and "bbox_xyxy_normalized" in reference:
                    normalized_box(reference["bbox_xyxy_normalized"])
            boxes = [
                reference.get("ref_box_xyxy") or reference.get("bbox_xyxy") or reference.get("bbox_xyxy_normalized")
                for reference in references
                if isinstance(reference, Mapping)
            ]
        else:
            boxes = [row.get("ref_box_xyxy")]
        if any(isinstance(box, list) and len(box) == 4 for box in boxes):
            eligible.append(case_id)
    return tuple(eligible)


def summarize_dino_rows(rows: Iterable[Mapping[str, object]], expected_case_ids: Sequence[str]) -> dict[str, Any]:
    """Macro-average exactly one finite DINO row per expected case."""
    expected = tuple(str(value) for value in expected_case_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_case_ids must be non-empty and unique")
    expected_set = set(expected)
    found: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        case_id = str(row.get("selected_case_id") or task_id(row))
        if case_id in expected_set:
            found.setdefault(case_id, []).append(row)
    missing = sorted(expected_set - set(found))
    duplicates = sorted(case_id for case_id, values in found.items() if len(values) != 1)
    if missing or duplicates:
        raise ValueError(f"missing={missing} duplicate={duplicates}")

    level_sums: dict[str, float] = {}
    level_counts: dict[str, int] = {}
    total = 0.0
    for case_id in expected:
        row = found[case_id][0]
        score = _finite(row["dino_score"], f"{case_id}/dino_score")
        level = normalize_memory_level(row.get("memory_level"))
        total += score
        level_sums[level] = level_sums.get(level, 0.0) + score
        level_counts[level] = level_counts.get(level, 0) + 1
    result: dict[str, Any] = {
        "cases": len(expected),
        "dino_overall": total / len(expected),
        "level_sums": level_sums,
        "level_counts": level_counts,
    }
    result.update({f"dino_{level}": level_sums[level] / level_counts[level] for level in sorted(level_sums)})
    return result


def summarize_dino_csv(path: Path, expected_case_ids: Sequence[str]) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        return summarize_dino_rows(csv.DictReader(handle), expected_case_ids)


def combine_dino_summaries(summaries: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Combine partitions by case count, never by unweighted partition mean."""
    if not summaries:
        raise ValueError("at least one DINO summary is required")
    cases = sum(int(row["cases"]) for row in summaries)
    if cases <= 0:
        raise ValueError("combined DINO summary must contain cases")
    total = sum(_finite(row["dino_overall"], "dino_overall") * int(row["cases"]) for row in summaries)
    level_sums: dict[str, float] = {}
    level_counts: dict[str, int] = {}
    for row in summaries:
        for level, value in dict(row["level_sums"]).items():
            level_sums[str(level)] = level_sums.get(str(level), 0.0) + _finite(value, f"{level}/sum")
        for level, value in dict(row["level_counts"]).items():
            level_counts[str(level)] = level_counts.get(str(level), 0) + int(value)
    result: dict[str, Any] = {
        "cases": cases,
        "dino_overall": total / cases,
        "level_sums": level_sums,
        "level_counts": level_counts,
    }
    result.update({f"dino_{level}": level_sums[level] / level_counts[level] for level in sorted(level_sums)})
    return result


def single_case_dino_metrics(score: float, memory_level: str) -> dict[str, float | None]:
    """Represent a case score without pretending it belongs to every subgroup."""
    value = _finite(score, "dino_score")
    level = normalize_memory_level(memory_level)
    return {
        "dino_overall": value,
        "dino_object": value if level == "object" else None,
        "dino_scene": value if level == "scene" else None,
    }
