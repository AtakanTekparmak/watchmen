"""Group G — slow-update fence helpers (plan §7l).

Coverage:
* Presence check (both markers / missing / duplicate).
* Half-fence (only START or only END) → raises.
* inject_slow_update_field places at END of SKILL.md.
* is_in_slow_update_region boundary membership.
* extract / replace round-trip.
"""

from __future__ import annotations

import pytest

from skill_evolve.shared.slow_update import (
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    extract_slow_update_field,
    has_slow_update_field,
    inject_slow_update_field,
    is_in_slow_update_region,
    replace_slow_update_field,
)


# ─── marker constants ────────────────────────────────────────────────────


def test_marker_strings_are_paper_faithful() -> None:
    assert SLOW_UPDATE_START == "<!-- SLOW_UPDATE_START -->"
    assert SLOW_UPDATE_END == "<!-- SLOW_UPDATE_END -->"


# ─── presence / well-formedness ─────────────────────────────────────────


def test_has_slow_update_field_both_present() -> None:
    text = f"# Skill\n\n{SLOW_UPDATE_START}\nhello\n{SLOW_UPDATE_END}\n"
    assert has_slow_update_field(text) is True


def test_has_slow_update_field_missing() -> None:
    assert has_slow_update_field("# Skill\n") is False


def test_has_slow_update_field_only_start() -> None:
    assert has_slow_update_field(f"# Skill\n{SLOW_UPDATE_START}\nhi") is False


def test_has_slow_update_field_only_end() -> None:
    assert has_slow_update_field(f"# Skill\n{SLOW_UPDATE_END}\n") is False


def test_has_slow_update_field_end_before_start() -> None:
    text = f"{SLOW_UPDATE_END}\nhi\n{SLOW_UPDATE_START}"
    assert has_slow_update_field(text) is False


def test_has_slow_update_field_duplicate_markers() -> None:
    text = (
        f"{SLOW_UPDATE_START}\na\n{SLOW_UPDATE_END}\n"
        f"{SLOW_UPDATE_START}\nb\n{SLOW_UPDATE_END}\n"
    )
    assert has_slow_update_field(text) is False


# ─── half-fence raises ──────────────────────────────────────────────────


def test_extract_half_fence_only_start_raises() -> None:
    text = f"# Skill\n{SLOW_UPDATE_START}\nstuff"
    with pytest.raises(ValueError, match="half-fence"):
        extract_slow_update_field(text)


def test_extract_half_fence_only_end_raises() -> None:
    text = f"# Skill\nstuff\n{SLOW_UPDATE_END}"
    with pytest.raises(ValueError, match="half-fence"):
        extract_slow_update_field(text)


def test_extract_no_fence_returns_empty() -> None:
    assert extract_slow_update_field("# Skill\n") == ""


# ─── inject places at END of SKILL.md ───────────────────────────────────


def test_inject_appends_at_end() -> None:
    body = "# Skill\n\nSome prose.\n"
    out = inject_slow_update_field(body, content="lessons")
    assert out.endswith(f"{SLOW_UPDATE_START}\nlessons\n{SLOW_UPDATE_END}\n")
    # bytes BEFORE the fence are body.rstrip() + "\n\n"
    assert out.startswith("# Skill\n\nSome prose.")


def test_inject_idempotent_double_inject_raises() -> None:
    body = inject_slow_update_field("# Skill\n", content="a")
    with pytest.raises(ValueError, match="already present"):
        inject_slow_update_field(body, content="b")


# ─── extract / replace round-trip ───────────────────────────────────────


def test_extract_replace_round_trip() -> None:
    body = inject_slow_update_field("# Skill\nprose\n", content="first")
    assert extract_slow_update_field(body) == "first"

    body2 = replace_slow_update_field(body, "second")
    assert extract_slow_update_field(body2) == "second"
    # Out-of-fence bytes preserved.
    head_orig = body.split(SLOW_UPDATE_START)[0]
    head_new = body2.split(SLOW_UPDATE_START)[0]
    assert head_orig == head_new


def test_replace_when_no_fence_present_injects() -> None:
    body = "# Skill\nprose\n"
    out = replace_slow_update_field(body, "guidance")
    assert has_slow_update_field(out)
    assert extract_slow_update_field(out) == "guidance"


def test_replace_when_half_fence_raises() -> None:
    body = f"# Skill\n{SLOW_UPDATE_START}\nhi"
    with pytest.raises(ValueError, match="half-fence"):
        replace_slow_update_field(body, "x")


# ─── boundary membership ───────────────────────────────────────────────


def test_is_in_slow_update_region_inside() -> None:
    body = inject_slow_update_field("# Skill\nprose\n", content="abc")
    start_idx = body.find(SLOW_UPDATE_START)
    end_idx = body.find(SLOW_UPDATE_END)
    middle = start_idx + len(SLOW_UPDATE_START) + 1  # past "\n"
    assert is_in_slow_update_region(body, middle) is True
    # Boundaries: offset AT the marker bytes is considered inside.
    assert is_in_slow_update_region(body, start_idx) is True
    assert is_in_slow_update_region(body, end_idx) is True


def test_is_in_slow_update_region_outside() -> None:
    body = inject_slow_update_field("# Skill\nprose\n", content="abc")
    # Offset 0 (before any fence) is out.
    assert is_in_slow_update_region(body, 0) is False
    # Offset past END marker is out.
    end_idx = body.find(SLOW_UPDATE_END) + len(SLOW_UPDATE_END)
    assert is_in_slow_update_region(body, end_idx) is False


def test_is_in_slow_update_region_no_fence() -> None:
    assert is_in_slow_update_region("# Skill\n", 3) is False
