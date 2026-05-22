"""Unit tests for daycare.anonymize (Stream 6.3)."""

from __future__ import annotations

from daycare.anonymize import AnonymizeContext, strip, tool_call_strip


def _ctx(**overrides) -> AnonymizeContext:
    base = dict(
        project_slugs={"ctf"},
        skill_slugs={"systematic-debugging"},
        source_repo="/Users/me/work/ctf",
        user_home="/Users/me",
    )
    base.update(overrides)
    return AnonymizeContext(**base)


def test_strip_uuid():
    ctx = _ctx()
    text = "session 9f2a8b1c-1234-5678-9abc-def012345678 logged in"
    out = strip(text, ctx)
    assert "9f2a8b1c" not in out
    assert "<SESSION_ID>" in out


def test_strip_home_path():
    """Home prefix replaced with /<USER>."""
    ctx = _ctx(source_repo="")  # disable source_repo so user_home runs cleanly
    text = "Open /Users/me/Documents/notes.md"
    out = strip(text, ctx)
    assert "/Users/me" not in out
    assert "/<USER>/Documents/notes.md" in out


def test_strip_source_repo():
    """source_repo replaced with <PROJECT_ROOT> BEFORE user_home runs
    (more-specific match wins)."""
    ctx = _ctx()
    text = "cd /Users/me/work/ctf && ls"
    out = strip(text, ctx)
    assert "/Users/me/work/ctf" not in out
    assert "<PROJECT_ROOT>" in out


def test_strip_skill_slug_whole_word():
    """Skill slugs replaced as whole words only."""
    ctx = _ctx()
    text = "Use the systematic-debugging skill now."
    out = strip(text, ctx)
    assert "systematic-debugging" not in out
    assert "<SKILL_SLUG>" in out


def test_strip_skill_slug_substring_safe():
    """Whole-word boundary protects substrings — 'ctf' should not match
    inside 'ctfsomething'."""
    ctx = _ctx(project_slugs={"ctf"}, skill_slugs=set())
    text = "ctfsomething is unrelated"
    out = strip(text, ctx)
    assert "ctfsomething" in out  # unchanged
    assert "<REPO_SLUG>" not in out


def test_strip_email():
    ctx = _ctx()
    out = strip("Contact: alice@example.com please", ctx)
    assert "alice@example.com" not in out
    assert "<EMAIL>" in out


def test_strip_or_key():
    ctx = _ctx()
    out = strip("key=sk-or-abc123XYZ-_456 use it", ctx)
    assert "sk-or-abc123" not in out
    assert "<OR_KEY>" in out


def test_strip_idempotent():
    ctx = _ctx()
    text = "/Users/me/work/ctf path here"
    out1 = strip(text, ctx)
    out2 = strip(out1, ctx)
    assert out1 == out2


def test_strip_non_string_passthrough():
    ctx = _ctx()
    # strip() returns the same object on non-string inputs (None / int).
    assert strip(None, ctx) is None
    assert strip("", ctx) == ""


# ─── tool_call_strip: nested dict handling ───────────────────────────────


def test_tool_call_strip_handles_nested_dicts():
    """MCP tool args often have nested structures — every string value
    inside the tree gets stripped."""
    ctx = _ctx()
    tc = {
        "name": "Bash",
        "input": {
            "command": "cd /Users/me/work/ctf && ls",
            "options": {
                "cwd": "/Users/me/work/ctf",
                "tags": ["systematic-debugging", "other"],
            },
        },
    }
    out = tool_call_strip(tc, ctx)
    # Original is not mutated (deep copy).
    assert tc["input"]["command"] == "cd /Users/me/work/ctf && ls"
    # Output is stripped everywhere.
    assert "/Users/me/work/ctf" not in out["input"]["command"]
    assert "<PROJECT_ROOT>" in out["input"]["command"]
    assert "<PROJECT_ROOT>" in out["input"]["options"]["cwd"]
    # List of strings handled too.
    assert out["input"]["options"]["tags"][0] == "<SKILL_SLUG>"
    assert out["input"]["options"]["tags"][1] == "other"


def test_tool_call_strip_preserves_non_string_scalars():
    ctx = _ctx()
    tc = {"name": "Bash", "input": {"timeout": 30, "verbose": True, "extra": None}}
    out = tool_call_strip(tc, ctx)
    assert out["input"]["timeout"] == 30
    assert out["input"]["verbose"] is True
    assert out["input"]["extra"] is None
