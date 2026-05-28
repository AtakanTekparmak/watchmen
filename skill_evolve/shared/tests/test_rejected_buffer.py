"""Unit tests for :mod:`skill_evolve.shared.rejected_buffer`.

Plan section E.7: cover the bounded ring eviction, JSONL round-trip,
and empty-vs-non-empty render. Module-level renderer fallback
(``(none yet)`` vs body) is exercised through ``track_a.prompts``
indirectly; the helper here is tested directly via the buffer's own
``render_for_prompt``.
"""

from __future__ import annotations

from pathlib import Path

from skill_evolve.shared.rejected_buffer import (
    RejectedBuffer,
    RejectedEdit,
    render_for_prompt,
)


def _edit(i: int, *, reason: str = "val_tie", patch: str = "x") -> RejectedEdit:
    return RejectedEdit(
        patch_text=patch,
        delta_train=0.0,
        delta_val=0.0,
        rejection_reason=reason,
        iteration=i,
    )


def test_bounded_ring_evicts_oldest() -> None:
    buf = RejectedBuffer(capacity=10)
    for i in range(15):  # push K+5 entries
        buf.push(_edit(i))
    assert len(buf) == 10
    recent = buf.recent(10)
    # First 5 must have been evicted (FIFO).
    iterations = [e.iteration for e in recent]
    assert iterations == list(range(5, 15))


def test_recent_returns_tail() -> None:
    buf = RejectedBuffer(capacity=10)
    for i in range(5):
        buf.push(_edit(i))
    assert [e.iteration for e in buf.recent(3)] == [2, 3, 4]
    assert buf.recent(0) == []
    # Asking for more than we have returns the whole tail.
    assert [e.iteration for e in buf.recent(99)] == [0, 1, 2, 3, 4]


def test_render_empty_returns_empty_string() -> None:
    buf = RejectedBuffer(capacity=5)
    assert buf.render_for_prompt() == ""
    # Module helper mirrors that behavior — caller decides on fallback.
    assert render_for_prompt(buf) == ""


def test_render_nonempty_contains_patch_head() -> None:
    buf = RejectedBuffer(capacity=3)
    buf.push(_edit(7, patch="ALPHA_PATCH_HEAD"))
    buf.push(_edit(8, patch="BETA_PATCH_HEAD"))
    rendered = buf.render_for_prompt()
    assert "ALPHA_PATCH_HEAD" in rendered
    assert "BETA_PATCH_HEAD" in rendered
    assert "iter 7" in rendered
    assert "iter 8" in rendered
    assert "val_tie" in rendered


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    buf = RejectedBuffer(capacity=4)
    for i in range(4):
        buf.push(
            RejectedEdit(
                patch_text=f"patch-{i}",
                delta_train=float(i) * 0.1,
                delta_val=-float(i) * 0.05,
                rejection_reason="val_not_strict_gt",
                iteration=i,
            )
        )
    out = tmp_path / "rejected.jsonl"
    buf.to_jsonl(out)
    text = out.read_text(encoding="utf-8").splitlines()
    assert len(text) == 4

    rehydrated = RejectedBuffer.from_jsonl(out, capacity=4)
    assert len(rehydrated) == 4
    items = rehydrated.recent(4)
    assert [e.iteration for e in items] == [0, 1, 2, 3]
    assert [e.patch_text for e in items] == [
        "patch-0",
        "patch-1",
        "patch-2",
        "patch-3",
    ]
    assert items[2].delta_train == 0.2
    assert items[2].delta_val == -0.1
    assert items[3].rejection_reason == "val_not_strict_gt"


def test_jsonl_roundtrip_skips_malformed_lines(tmp_path: Path) -> None:
    out = tmp_path / "rejected.jsonl"
    out.write_text(
        '{"patch_text":"a","delta_train":0,"delta_val":0,'
        '"rejection_reason":"val_tie","iteration":1}\n'
        "this-is-not-json\n"
        '{"patch_text":"b","delta_train":0,"delta_val":0,'
        '"rejection_reason":"val_tie","iteration":2}\n',
        encoding="utf-8",
    )
    buf = RejectedBuffer.from_jsonl(out, capacity=10)
    assert [e.iteration for e in buf.recent(10)] == [1, 2]


def test_render_helper_proxies_buffer() -> None:
    """The module-level ``render_for_prompt`` mirrors ``buffer.render_for_prompt``."""
    buf = RejectedBuffer(capacity=2)
    buf.push(_edit(42, patch="SENTINEL"))
    assert render_for_prompt(buf) == buf.render_for_prompt()
    assert "SENTINEL" in render_for_prompt(buf)
