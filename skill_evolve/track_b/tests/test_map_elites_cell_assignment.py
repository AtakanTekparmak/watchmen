"""Tests for MAP-Elites feature binning."""

from __future__ import annotations

from skill_evolve.track_b.openevolve_skills.database import (
    FEATURE_BIN_EDGES,
    assign_bin,
    cell_key,
    compute_features,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import FolderArtifact


def _make_skill(name: str, desc: str, body_chars: int = 100) -> tuple[str, str]:
    body = "x" * body_chars
    content = f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n{body}\n"
    return f"{name}/SKILL.md", content


def test_assign_bin_boundaries():
    edges = FEATURE_BIN_EDGES["num_skills"]  # [4, 8, 13, inf]
    # 3 skills → bin 0 (<4).
    assert assign_bin(3, edges) == 0
    assert assign_bin(4, edges) == 1  # value == 4 → bin 1
    assert assign_bin(7, edges) == 1
    assert assign_bin(8, edges) == 2
    assert assign_bin(20, edges) == 3


def test_total_tokens_bin_edges():
    edges = FEATURE_BIN_EDGES["total_tokens"]
    assert assign_bin(500, edges) == 0
    assert assign_bin(1500, edges) == 1
    assert assign_bin(5000, edges) == 2
    assert assign_bin(8000, edges) == 3
    assert assign_bin(100000, edges) == 4


def test_compute_features_small_folder():
    files = dict([
        _make_skill("a", "short desc", body_chars=50),
        _make_skill("b", "short desc", body_chars=50),
    ])
    art = FolderArtifact(files=files)
    feats = compute_features(art)
    assert feats["num_skills"] == 2
    assert feats["total_tokens"] > 0
    # Short description ≤ 120 chars → specificity bin 0.
    assert feats["avg_specificity"] < 120


def test_cell_key_is_deterministic():
    art = FolderArtifact(files=dict([
        _make_skill("a", "x" * 60),
        _make_skill("b", "x" * 60),
    ]))
    k1 = cell_key(art)
    k2 = cell_key(art)
    assert k1 == k2
    # Tuple length = number of feature dims.
    assert len(k1) == len(FEATURE_BIN_EDGES)


def test_cell_key_shifts_with_more_skills():
    small = FolderArtifact(files=dict([
        _make_skill(f"s{i}", "desc", 30) for i in range(2)
    ]))
    big = FolderArtifact(files=dict([
        _make_skill(f"s{i}", "desc", 30) for i in range(10)
    ]))
    assert cell_key(small)[0] < cell_key(big)[0]


def test_specificity_bin_differs_by_description_length():
    short = FolderArtifact(files=dict([_make_skill("a", "short")]))
    longd = FolderArtifact(files=dict([_make_skill("a", "x" * 300)]))
    f_short = compute_features(short)
    f_long = compute_features(longd)
    assert f_short["avg_specificity"] < f_long["avg_specificity"]
    # The two folders should land in different specificity bins.
    edges = FEATURE_BIN_EDGES["avg_specificity"]
    assert assign_bin(f_short["avg_specificity"], edges) != \
        assign_bin(f_long["avg_specificity"], edges)
