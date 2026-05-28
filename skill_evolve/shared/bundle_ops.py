"""Bundle ops — apply / validate / hash / token-count for skill bundles.

Ported from ``watchmen-fukara/daycare/src/daycare/mutator.py`` (apply,
shebang_insurance, validate_scripts, hash_bundle, parse_and_apply) and
``daycare/src/daycare/evolve.py`` lines 66-115 (_bundle_tokens →
``bundle_tokens``; ``list_scripts``).

Constants (locked cross-group contract — see plan section 7a):
    MAX_SKILL_TOKENS   = 3000
    MAX_BUNDLE_TOKENS  = 60000
    APPLE_DOUBLE_PREFIX = "._"

The macOS AppleDouble prefix filter is applied INLINE in ``bundle_tokens``
and ``list_scripts`` so uploads of bundles that already contain ``._foo.py``
metadata files (a common pain point when shuttling tarballs through macOS)
are tolerated transparently.
"""

from __future__ import annotations

import hashlib
import py_compile
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from skill_evolve.shared.patch_parser import (
    FileOp,
    SentinelParseError,
    parse_sentinel_blocks,
)


# ─── Constants ────────────────────────────────────────────────────────────

MAX_SKILL_TOKENS = 3000
MAX_BUNDLE_TOKENS = 60000  # 2× largest known seed (~30k); bounds script growth
APPLE_DOUBLE_PREFIX = "._"


# ─── Exceptions + result types ────────────────────────────────────────────


class BundleRejected(Exception):
    """Raised by ``SkillFolder.write()`` (and friends) when a candidate
    bundle fails the smoke gate or exceeds a token cap. Caught in the
    runner's pass loop to record a rejection without an eval call.
    """


@dataclass
class SmokeResult:
    """Result of ``validate_scripts``.

    Fields:
        ok: True iff every ``.py`` py_compiles and every ``.sh`` passes
            ``bash -n``.
        failures: list of (path, message) tuples. Collected — not
            short-circuited — so the proposer can see EVERY broken file
            in a single pass.
    """

    ok: bool
    failures: list[tuple[Path, str]] = field(default_factory=list)


@dataclass
class ApplyResult:
    """Result of ``apply_file_ops``."""

    candidate_dir: Path
    applied: list[FileOp] = field(default_factory=list)


# ─── Token counting ───────────────────────────────────────────────────────


def bundle_tokens(bundle_dir: Path) -> int:
    """Sum cl100k_base tokens across SKILL.md + scripts/**/*.{py,sh} +
    references/**/*.md. Hidden files (dotfiles), AppleDouble ``._*``
    files and ``__pycache__`` are excluded.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        # Soft fallback for environments without tiktoken: return bytes
        # / 4 as a rough proxy. The hard caps in the proposer prompt
        # are still expressed in token units; this fallback only kicks
        # in for unit tests / CI.
        return _bundle_bytes_approx_tokens(bundle_dir)

    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    patterns: list[Path] = [
        bundle_dir / "SKILL.md",
        *bundle_dir.glob("scripts/**/*.py"),
        *bundle_dir.glob("scripts/**/*.sh"),
        *bundle_dir.glob("references/**/*.md"),
        # Bundles may also nest SKILL.md per-skill — count those too.
        *bundle_dir.glob("*/SKILL.md"),
    ]
    seen: set[Path] = set()
    for p in patterns:
        if p in seen:
            continue
        seen.add(p)
        if not p.is_file():
            continue
        if p.name.startswith(APPLE_DOUBLE_PREFIX):
            continue
        if p.name.startswith("."):
            continue
        if "__pycache__" in str(p):
            continue
        try:
            total += len(enc.encode(p.read_text(encoding="utf-8", errors="ignore")))
        except Exception:  # noqa: BLE001 — counting is best-effort
            pass
    return total


def _bundle_bytes_approx_tokens(bundle_dir: Path) -> int:
    """Fallback when tiktoken is unavailable: bytes // 4 ≈ tokens."""
    total_bytes = 0
    for p in bundle_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(APPLE_DOUBLE_PREFIX) or p.name.startswith("."):
            continue
        if "__pycache__" in str(p):
            continue
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass
    return total_bytes // 4


def list_scripts(parent_bundle: Path) -> str:
    """Return the parent bundle's scripts/ directory layout with byte sizes.

    AppleDouble ``._*`` metadata files are excluded.
    """
    scripts_dir = parent_bundle / "scripts"
    if not scripts_dir.exists():
        return "(no scripts/ in parent bundle)"
    lines: list[str] = []
    for f in sorted(scripts_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.name.startswith(APPLE_DOUBLE_PREFIX):
            continue
        if f.name.startswith("."):
            continue
        rel = f.relative_to(parent_bundle)
        lines.append(f"{rel}  ({f.stat().st_size} bytes)")
    return "\n".join(lines) if lines else "(scripts/ is empty)"


# ─── Apply ops to a candidate dir ─────────────────────────────────────────


def apply_file_ops(
    parent_bundle_dir: Path,
    ops: list[FileOp],
    candidate_dir: Path,
) -> ApplyResult:
    """Copy ``parent_bundle_dir`` to ``candidate_dir``, then apply ``ops``.

    ADD_FILE writes a new file (parent dirs created); EDIT_FILE overwrites
    in place; DELETE_FILE removes the file (silently if it vanished
    underneath us); REWRITE_FOLDER blows away the named folder and
    rewrites it from the ``files`` mapping.

    Returns an ``ApplyResult`` carrying the candidate dir + the applied
    op list (for logging).
    """
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    shutil.copytree(parent_bundle_dir, candidate_dir)

    for op in ops:
        target = candidate_dir / op.path if op.path else candidate_dir

        if op.op == "ADD_FILE":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(op.content or "", encoding="utf-8")
        elif op.op == "EDIT_FILE":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(op.content or "", encoding="utf-8")
        elif op.op == "DELETE_FILE":
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        elif op.op == "REWRITE_FOLDER":
            # Daycare-style: ``files`` carries a {relpath: content} mapping
            # under a named folder. track_b-style with a serialized blob
            # is NOT handled here — track_b's own apply_patch handles
            # FolderArtifact materialization.
            if op.files is None:
                continue
            folder = candidate_dir / op.path
            if folder.exists():
                if folder.is_dir():
                    shutil.rmtree(folder)
                else:
                    folder.unlink()
            folder.mkdir(parents=True, exist_ok=True)
            for rel, content in op.files.items():
                fpath = candidate_dir / rel
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content, encoding="utf-8")

    return ApplyResult(candidate_dir=candidate_dir, applied=list(ops))


# ─── Shebang insurance (K7) ───────────────────────────────────────────────


def shebang_insurance(file_path: Path) -> None:
    """Ensure .py / .sh files have the correct shebang at line 1.

    Idempotent — if the shebang is already present (exact match), nothing
    happens. Otherwise the missing line is prepended.
    """
    if not file_path.exists() or not file_path.is_file():
        return

    suffix = file_path.suffix.lower()
    if suffix == ".py":
        wanted = "#!/usr/bin/env python3"
    elif suffix == ".sh":
        wanted = "#!/bin/bash"
    else:
        return

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    first_line = text.split("\n", 1)[0] if text else ""
    if first_line.strip() == wanted:
        return

    new_text = wanted + "\n" + text
    try:
        file_path.write_text(new_text, encoding="utf-8")
    except OSError:
        return


# ─── Script syntax validation ─────────────────────────────────────────────


def validate_scripts(
    candidate_dir: Path,
    parent_bundle_dir: Path | None = None,
) -> SmokeResult:
    """Run ``py_compile`` on every .py and ``bash -n`` on every .sh under
    ``candidate_dir``. Returns a :class:`SmokeResult`.

    If ``parent_bundle_dir`` is provided, scripts that are byte-identical
    to the parent are skipped — we only validate files that the mutation
    actually changed. This prevents pre-existing broken scripts in the
    parent bundle from failing every candidate.

    ALL failures are collected (no short-circuit) so the proposer sees
    every broken file at once.
    """
    failures: list[tuple[Path, str]] = []

    for fpath in candidate_dir.rglob("*"):
        if not fpath.is_file():
            continue
        # Skip AppleDouble metadata and dotfiles.
        if fpath.name.startswith(APPLE_DOUBLE_PREFIX):
            continue
        if fpath.name.startswith("."):
            continue
        if "__pycache__" in fpath.parts:
            continue

        rel = fpath.relative_to(candidate_dir)

        # Skip if unchanged from parent — don't fail candidates for
        # pre-existing breakage they didn't introduce.
        if parent_bundle_dir is not None:
            parent_copy = parent_bundle_dir / rel
            if parent_copy.exists():
                try:
                    if fpath.read_bytes() == parent_copy.read_bytes():
                        continue
                except OSError:
                    pass

        suffix = fpath.suffix.lower()
        if suffix == ".py":
            try:
                py_compile.compile(str(fpath), doraise=True)
            except py_compile.PyCompileError as exc:
                # Truncate the message so a long traceback doesn't blow
                # out the history.jsonl row.
                msg = str(exc).strip()[:200]
                failures.append((rel, f"py_compile:{msg}"))
            except (OSError, ValueError) as exc:
                failures.append((rel, f"py_compile_subprocess_fail:{exc}"))
        elif suffix == ".sh":
            try:
                proc = subprocess.run(
                    ["bash", "-n", str(fpath)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                failures.append((rel, f"bash_n_subprocess_fail:{exc}"))
                continue
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()[:200]
                failures.append((rel, f"bash_n:{err}"))

    return SmokeResult(ok=not failures, failures=failures)


# ─── Hashing for genealogy / dedup ────────────────────────────────────────


def hash_bundle(bundle_dir: Path) -> str:
    """SHA256 of every file's relative path + content under ``bundle_dir``.

    Files are sorted by relative path; for each file we hash the path bytes
    then the raw content bytes. The result is deterministic across runs
    and across machines (no mtimes / inodes involved). AppleDouble
    metadata files are excluded.
    """
    h = hashlib.sha256()
    if not bundle_dir.exists():
        return h.hexdigest()

    files = sorted(
        p
        for p in bundle_dir.rglob("*")
        if p.is_file() and not p.name.startswith(APPLE_DOUBLE_PREFIX)
    )
    for fpath in files:
        rel = fpath.relative_to(bundle_dir).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        try:
            h.update(fpath.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")

    return h.hexdigest()


# ─── Convenience wrapper ──────────────────────────────────────────────────


def parse_and_apply(
    text: str,
    parent_bundle_dir: Path,
    candidate_dir: Path,
) -> tuple[Path, list[tuple[Path, str]]]:
    """Parse → apply → shebang-fix → validate. Returns (candidate_dir, failures).

    Existing-path set is built from ``parent_bundle_dir`` so ADD/EDIT/DELETE
    existence rules fire. Shebang insurance runs after apply so newly-added
    scripts inherit a shebang even if the proposer omitted one.
    """
    existing_paths = {
        p.relative_to(parent_bundle_dir).as_posix()
        for p in parent_bundle_dir.rglob("*")
        if p.is_file() and not p.name.startswith(APPLE_DOUBLE_PREFIX)
    }

    ops = parse_sentinel_blocks(text, existing_paths=existing_paths, strict=True)
    apply_file_ops(parent_bundle_dir, ops, candidate_dir)

    # Shebang insurance on every script that landed in the candidate dir.
    for fpath in candidate_dir.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in (".py", ".sh"):
            shebang_insurance(fpath)

    smoke = validate_scripts(candidate_dir, parent_bundle_dir=parent_bundle_dir)
    return candidate_dir, smoke.failures


# ─── Deployment-strip copy (Group G, 2026-05-28; plan §7l + §G.9) ────────


# Files / directories that are NEVER deployed at inference. Plan §7l locks
# ``meta_skill.md`` as training-only — it lives at the bundle root as the
# consolidator's audit log but the inference-side agent must not see it.
# AppleDouble metadata and __pycache__/ are excluded for the same reasons
# they're filtered from ``bundle_tokens`` / ``list_scripts``.
_DEPLOY_EXCLUDE_ROOT_FILES = frozenset({"meta_skill.md"})
_DEPLOY_EXCLUDE_DIR_NAMES = frozenset({"__pycache__"})


def copy_for_deployment(src: Path, dst: Path) -> None:
    """Copy a bundle from ``src`` to ``dst`` minus training-only files.

    Excludes:
        * ``meta_skill.md`` (bundle root) — consolidator audit log, plan §7l.
        * ``._*`` (macOS AppleDouble metadata).
        * ``__pycache__/`` directories.
        * ``.pyc`` files anywhere in the tree.

    Used at every bundle-promotion / validation-eval copy site so the
    deployed bundle never carries training-side bookkeeping. The
    ``meta_skill.md`` exclusion is paper-faithful (plan §7l): the file
    lives ONLY in the evolution sandbox and is regenerated per-run.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"copy_for_deployment: src not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(dirpath: str, names: list[str]) -> list[str]:
        skip: list[str] = []
        is_root = Path(dirpath).resolve() == src.resolve()
        for name in names:
            if name in _DEPLOY_EXCLUDE_DIR_NAMES:
                skip.append(name)
                continue
            if name.startswith(APPLE_DOUBLE_PREFIX):
                skip.append(name)
                continue
            if name.endswith(".pyc"):
                skip.append(name)
                continue
            if is_root and name in _DEPLOY_EXCLUDE_ROOT_FILES:
                skip.append(name)
                continue
        return skip

    shutil.copytree(src, dst, ignore=_ignore)


# Silence unused-import diagnostics for ``sys`` / ``SentinelParseError`` — they
# may be referenced from re-exports / future helpers.
_ = sys
_ = SentinelParseError


__all__ = [
    "MAX_SKILL_TOKENS",
    "MAX_BUNDLE_TOKENS",
    "APPLE_DOUBLE_PREFIX",
    "BundleRejected",
    "SmokeResult",
    "ApplyResult",
    "bundle_tokens",
    "list_scripts",
    "apply_file_ops",
    "shebang_insurance",
    "validate_scripts",
    "hash_bundle",
    "parse_and_apply",
    "copy_for_deployment",
]
