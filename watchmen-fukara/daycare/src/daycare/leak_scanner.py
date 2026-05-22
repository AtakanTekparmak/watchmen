"""Output-side eval-identifier leak scanner (K3 + failure-mode #5 defense).

Every candidate bundle (Phase 3b.5) is scanned for verbatim leaks of any
eval identifier. The fingerprint set is built from ``eval_set.jsonl`` +
``projects.json`` at run start; the scanner walks the candidate dir
line-by-line and flags any case-sensitive substring match.

Policy:
  - ``zero`` (default): any leak → reject the candidate, log to history.
  - ``warn``: log to ``leak_log.md``, candidate proceeds.

n-gram defense (failure mode #5): identifiers that show up in ≥2 different
prompts but aren't in projects.json (e.g. external GitHub repo slugs the
anonymizer's regex set missed) are still caught by cross-eval recurrence.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# ─── Dataclass ────────────────────────────────────────────────────────────


@dataclass
class Leak:
    """One detected leak occurrence.

    Fields:
        file: relative path of the offending file inside the candidate dir.
        line: 1-indexed line number where the fingerprint was found.
        fingerprint: the literal string that matched.
        context: the offending line (trimmed) for human review.
    """

    file: str
    line: int
    fingerprint: str
    context: str


# ─── Fingerprint set construction ─────────────────────────────────────────

# Absolute paths appearing in prompts: anything starting with "/" up to the
# next whitespace or quote. We're permissive on what counts as a path char.
_ABS_PATH_RE = re.compile(r"/[A-Za-z0-9_\-./]+")

# OR API key pattern (also caught by anonymize.py, but we double-check on
# the output side per K3).
_OR_KEY_RE = re.compile(r"sk-or-[A-Za-z0-9\-_]{20,}")


def _extract_ngrams(text: str, n_min: int = 12, n_max: int = 30) -> set[str]:
    """Extract candidate n-grams of length n_min..n_max from ``text``.

    The naive O(n × (n_max - n_min)) sliding-window over every prompt would
    blow up memory for the cross-eval-recurrence pass — instead we extract
    only n-grams that look identifier-like: contiguous runs of
    ``[A-Za-z0-9_\\-/.]`` (the chars that show up in slugs, repo paths, and
    UUIDs minus the dash separators). This collapses arbitrary English
    prose down to the actual identifier surface.

    For each such run we emit substrings of length n_min..min(len, n_max).
    """
    grams: set[str] = set()
    # Find every identifier-like run.
    for run_match in re.finditer(r"[A-Za-z0-9_\-/.]{12,}", text):
        run = run_match.group(0)
        L = len(run)
        # Cap the upper bound at the run length.
        upper = min(L, n_max)
        for length in range(n_min, upper + 1):
            for start in range(0, L - length + 1):
                grams.add(run[start : start + length])
    return grams


def build_fingerprint_set(eval_set_path: Path, projects_json_path: Path) -> set[str]:
    """Build the leak-detection fingerprint set.

    Sources (per spec §"Phase 3b.5" and failure mode #5):
      1. Every ``source_session`` ID in eval_set.jsonl.
      2. Every absolute path (``/...``) appearing in any ``prompt``.
      3. Every 12–30 char n-gram appearing verbatim in ≥2 different
         ``prompt`` fields (cross-eval recurrence — catches identifiers
         the anonymizer missed).
      4. Project slugs + source_repo values from projects.json.
      5. OR API key patterns (``sk-or-...``).

    Returns a set of unique strings ≥ 4 chars (anything shorter is
    statistical noise — single tokens like "ctf" would false-positive).
    """
    fingerprints: set[str] = set()

    # ── Eval set pass ────────────────────────────────────────────────
    prompts: list[str] = []
    if eval_set_path.exists():
        try:
            with eval_set_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue

                    sess = row.get("source_session")
                    if isinstance(sess, str) and sess:
                        fingerprints.add(sess)

                    prompt = row.get("prompt")
                    if isinstance(prompt, str) and prompt:
                        prompts.append(prompt)
                        # Absolute paths in prompts.
                        for m in _ABS_PATH_RE.finditer(prompt):
                            p = m.group(0)
                            if len(p) >= 4:
                                fingerprints.add(p)
                        # OR keys (rare but catastrophic).
                        for m in _OR_KEY_RE.finditer(prompt):
                            fingerprints.add(m.group(0))
        except OSError:
            pass

    # ── Cross-eval n-gram recurrence pass ─────────────────────────────
    # An n-gram counts toward leak only if it appears in ≥2 DIFFERENT
    # prompt fields — within-prompt repetition is fine.
    if len(prompts) >= 2:
        gram_counts: Counter[str] = Counter()
        for prompt in prompts:
            grams = _extract_ngrams(prompt)
            for gram in grams:
                gram_counts[gram] += 1
        for gram, count in gram_counts.items():
            if count >= 2:
                fingerprints.add(gram)

    # ── projects.json pass ────────────────────────────────────────────
    if projects_json_path.exists():
        try:
            data = json.loads(projects_json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(key, str) and len(key) >= 3:
                        fingerprints.add(key)
                    if isinstance(val, dict):
                        sr = val.get("source_repo")
                        if isinstance(sr, str) and len(sr) >= 4:
                            fingerprints.add(sr)
                            # Plus the basename — "ctf" is too short but
                            # a longer basename (e.g. "synthetic-rl") is
                            # exactly the identifier surface we want.
                            base = Path(sr).name
                            if len(base) >= 4:
                                fingerprints.add(base)
        except (json.JSONDecodeError, OSError):
            pass

    # Drop empty/short entries to suppress false positives.
    return {fp for fp in fingerprints if fp and len(fp) >= 4}


# ─── Scanner ──────────────────────────────────────────────────────────────


def scan(candidate_dir: Path, fingerprints: set[str]) -> list[Leak]:
    """Walk every file under ``candidate_dir``; flag every fingerprint hit.

    Case-sensitive substring match per the spec. One Leak per (file, line,
    fingerprint) tuple — multiple fingerprints on the same line yield
    multiple Leak entries.

    Binary files (read raises UnicodeDecodeError) are skipped.
    """
    leaks: list[Leak] = []
    if not candidate_dir.exists() or not fingerprints:
        return leaks

    # Sort fingerprints by length descending so the first match found per
    # line is the most-specific one (cosmetic — every hit gets logged
    # regardless).
    sorted_fps = sorted(fingerprints, key=len, reverse=True)

    for fpath in candidate_dir.rglob("*"):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(candidate_dir).as_posix()
        try:
            text = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for fp in sorted_fps:
                if fp in line:
                    leaks.append(
                        Leak(
                            file=rel,
                            line=lineno,
                            fingerprint=fp,
                            context=line.strip()[:200],
                        )
                    )

    return leaks


# ─── Policy enforcement ───────────────────────────────────────────────────


def _render_leak_log(leaks: list[Leak]) -> str:
    """Render leaks as a markdown report (one section per file)."""
    if not leaks:
        return "# Leak log\n\n_No leaks detected._\n"

    lines = ["# Leak log", ""]
    # Group by file for readability.
    by_file: dict[str, list[Leak]] = {}
    for lk in leaks:
        by_file.setdefault(lk.file, []).append(lk)
    for file, entries in sorted(by_file.items()):
        lines.append(f"## {file}")
        lines.append("")
        lines.append("| Line | Fingerprint | Context |")
        lines.append("|---:|---|---|")
        for e in entries:
            # Escape pipe chars in context so the markdown table doesn't
            # break on awkward lines.
            ctx = e.context.replace("|", "\\|")
            fp = e.fingerprint.replace("|", "\\|")
            lines.append(f"| {e.line} | `{fp}` | `{ctx}` |")
        lines.append("")
    return "\n".join(lines) + "\n"


def enforce_policy(
    leaks: list[Leak],
    policy: str,
    leak_log_path: Path | None = None,
) -> bool:
    """Apply the leak policy. Returns True iff the candidate should be rejected.

    Args:
        leaks: scanner output.
        policy: ``zero`` rejects on any leak; ``warn`` only logs.
        leak_log_path: where to append the markdown report (warn policy or
            zero-with-leaks). Best-effort — failures here never affect the
            rejection decision.

    Returns:
        True → reject the candidate. False → continue scoring.
    """
    if leak_log_path is not None and leaks:
        try:
            leak_log_path.parent.mkdir(parents=True, exist_ok=True)
            leak_log_path.write_text(_render_leak_log(leaks), encoding="utf-8")
        except OSError:
            pass

    if policy == "zero":
        return bool(leaks)
    if policy == "warn":
        # Warn policy always proceeds — the log above is the side effect.
        return False
    # Unknown policy: behave like zero (safer default).
    return bool(leaks)
