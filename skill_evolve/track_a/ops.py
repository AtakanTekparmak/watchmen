"""Mutation operators for a SkillFolder.

Each operator is a pure function ``apply(folder, client, ...) -> SkillFolder``
that takes an *immutable-feeling* SkillFolder (we always ``.clone()``
first) and returns a new folder with the mutation applied. LLM calls go
through :class:`~skill_evolve.track_a.llm.LLMClient` so ``--force-synthetic``
drives deterministic test-time behavior.

The operator set matches the brief:

* ``AddSkill``            — new skill folder, LLM authors body.
* ``RemoveSkill``         — delete a skill folder.
* ``RenameSkill``         — rename folder + frontmatter.name.
* ``SplitSkill``          — one skill into two (LLM writes both bodies).
* ``MergeSkills``         — two skills into one (LLM writes merged body).
* ``RewriteSkillContent`` — rewrite one skill's body conditioned on critique
  and benchmark failures attributed to that skill.

``pick_op`` wraps an LLM planning call with a rule-based fallback, so a
mangled JSON reply doesn't break the loop.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .folder import SkillDoc, SkillFolder, make_skill_doc
from .llm import LLMClient
from .prompts import (
    AUTHOR_B_SYSTEM,
    AUTHOR_SCRIPT_SYSTEM,
    DESCRIBE_GOAL,
    MERGE_PROMPT,
    NEW_BODY_PROMPT,
    NEW_SCRIPT_PROMPT,
    OP_PLANNER_PROMPT,
    OP_PLANNER_SYSTEM,
    REWRITE_BODY_PROMPT,
    REWRITE_SCRIPT_PROMPT,
    SPLIT_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Op records (for logging / testability)
# ---------------------------------------------------------------------------

@dataclass
class OpRecord:
    op: str
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "args": self.args}


# ---------------------------------------------------------------------------
# JSON parsing helpers (LLMs love to add fences)
# ---------------------------------------------------------------------------

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(s: str) -> str:
    return _JSON_FENCE.sub("", s).strip()


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = _strip_fences(text)
    # Try direct parse, then substring from first "{" to last "}".
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Individual operators
# ---------------------------------------------------------------------------

def apply_add_skill(
    folder: SkillFolder,
    client: LLMClient,
    *,
    name: str,
    description: str,
    seed: str,
) -> SkillFolder:
    out = folder.clone()
    if out.by_name(name):
        raise ValueError(f"AddSkill: '{name}' already exists")
    body = client.complete(
        AUTHOR_B_SYSTEM,
        NEW_BODY_PROMPT.format(
            goal=DESCRIBE_GOAL,
            folder_name=name,
            description=description,
            seed=seed,
            folder_summary=out.render_summary(body_chars=400),
        ),
        tag="new_body",
    ).strip()
    doc = make_skill_doc(name, description=description, body=body)
    out.add(doc)
    return out


def apply_remove_skill(folder: SkillFolder, client: LLMClient, *, name: str) -> SkillFolder:
    out = folder.clone()
    if not out.remove(name):
        raise KeyError(f"RemoveSkill: '{name}' not in folder")
    return out


def apply_rename_skill(
    folder: SkillFolder,
    client: LLMClient,
    *,
    old: str,
    new: str,
) -> SkillFolder:
    out = folder.clone()
    out.rename(old, new)
    return out


def apply_split_skill(
    folder: SkillFolder,
    client: LLMClient,
    *,
    name: str,
    into: List[Dict[str, str]],
    rationale: str,
) -> SkillFolder:
    out = folder.clone()
    src = out.by_name(name)
    if not src:
        raise KeyError(f"SplitSkill: '{name}' not in folder")
    if len(into) != 2:
        raise ValueError("SplitSkill: 'into' must have exactly 2 entries")
    name_a, desc_a = into[0]["name"], into[0].get("description", "")
    name_b, desc_b = into[1]["name"], into[1].get("description", "")
    text = client.complete(
        AUTHOR_B_SYSTEM,
        SPLIT_PROMPT.format(
            goal=DESCRIBE_GOAL,
            frontmatter=json.dumps(src.frontmatter, indent=2),
            body=src.body,
            rationale=rationale,
            name_a=name_a, desc_a=desc_a,
            name_b=name_b, desc_b=desc_b,
        ),
        tag="split",
    )
    body_a, body_b = _parse_split_response(text)
    if body_a is None or body_b is None:
        raise ValueError("SplitSkill: could not parse split response")
    out.remove(name)
    out.add(make_skill_doc(name_a, description=desc_a, body=body_a))
    out.add(make_skill_doc(name_b, description=desc_b, body=body_b))
    return out


def _parse_split_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    m_a = re.search(r"<<<SKILL_A>>>(.*?)<<<END_SKILL_A>>>", text, re.DOTALL)
    m_b = re.search(r"<<<SKILL_B>>>(.*?)<<<END_SKILL_B>>>", text, re.DOTALL)
    a = m_a.group(1).strip() if m_a else None
    b = m_b.group(1).strip() if m_b else None
    return a, b


def apply_merge_skills(
    folder: SkillFolder,
    client: LLMClient,
    *,
    a: str,
    b: str,
    into: str,
    description: str,
    rationale: str,
) -> SkillFolder:
    out = folder.clone()
    doc_a = out.by_name(a)
    doc_b = out.by_name(b)
    if not doc_a or not doc_b:
        raise KeyError(f"MergeSkills: missing operand ({a} or {b})")
    text = client.complete(
        AUTHOR_B_SYSTEM,
        MERGE_PROMPT.format(
            goal=DESCRIBE_GOAL,
            frontmatter_a=json.dumps(doc_a.frontmatter, indent=2),
            body_a=doc_a.body,
            frontmatter_b=json.dumps(doc_b.frontmatter, indent=2),
            body_b=doc_b.body,
            rationale=rationale,
            into=into,
        ),
        tag="merge",
    ).strip()
    out.remove(a)
    out.remove(b)
    # 'into' may equal one of the originals; if so, it's now free.
    out.add(make_skill_doc(into, description=description, body=text))
    return out


# ---------------------------------------------------------------------------
# Script operators (Phase 2, 2026-04-22)
# ---------------------------------------------------------------------------

_SCRIPT_EXT_SHEBANG = {
    ".sh": "#!/usr/bin/env bash\nset -euo pipefail\n",
    ".py": "#!/usr/bin/env python3\n",
}


def _validate_script_path(path: str) -> str:
    """Gate for AddScript / RewriteScript / RemoveScript paths.

    Must be POSIX-relative, under ``scripts/``, and use a recognised
    executable extension. No traversal, no absolute paths.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("script path must be a non-empty string")
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError(f"script path must be relative and traversal-free: {path!r}")
    parts = path.split("/")
    if parts[0] != "scripts" or len(parts) < 2 or parts[-1] == "":
        raise ValueError(
            f"script path must start with 'scripts/' and name a file, got {path!r}"
        )
    suffix = path[path.rfind("."):] if "." in parts[-1] else ""
    if suffix not in _SCRIPT_EXT_SHEBANG:
        raise ValueError(
            f"script path must end in .sh or .py, got {path!r}"
        )
    return path


def _ensure_shebang(path: str, content: str) -> str:
    """Prefix a shebang if the content doesn't already start with one.

    LLMs reliably forget shebangs on .py and occasionally on .sh; a
    missing shebang means the agent has to know to invoke via
    ``python3`` / ``bash``. Cheap insurance.
    """
    if content.startswith("#!"):
        return content
    suffix = path[path.rfind("."):]
    header = _SCRIPT_EXT_SHEBANG.get(suffix, "")
    return header + content if header else content


def apply_add_script(
    folder: SkillFolder,
    client: LLMClient,
    *,
    skill: str,
    path: str,
    purpose: str,
) -> SkillFolder:
    out = folder.clone()
    doc = out.by_name(skill)
    if not doc:
        raise KeyError(f"AddScript: skill '{skill}' not in folder")
    _validate_script_path(path)
    if path in doc.auxiliary_files:
        raise ValueError(
            f"AddScript: '{path}' already exists in skill '{skill}'; "
            "use RewriteScript instead"
        )
    raw = client.complete(
        AUTHOR_SCRIPT_SYSTEM,
        NEW_SCRIPT_PROMPT.format(
            goal=DESCRIBE_GOAL,
            skill_name=skill,
            skill_description=doc.description,
            skill_body=doc.body[:1500],
            path=path,
            purpose=purpose,
        ),
        tag="new_script",
        max_tokens=2000,
    ).strip()
    # Strip accidental fences (LLMs do this even when told not to).
    raw = _JSON_FENCE.sub("", raw).strip()
    doc.auxiliary_files[path] = _ensure_shebang(path, raw) + (
        "\n" if not raw.endswith("\n") else ""
    )
    return out


def apply_rewrite_script(
    folder: SkillFolder,
    client: LLMClient,
    *,
    skill: str,
    path: str,
    critique: str,
) -> SkillFolder:
    out = folder.clone()
    doc = out.by_name(skill)
    if not doc:
        raise KeyError(f"RewriteScript: skill '{skill}' not in folder")
    _validate_script_path(path)
    if path not in doc.auxiliary_files:
        raise KeyError(
            f"RewriteScript: '{path}' not in skill '{skill}'"
        )
    current = doc.auxiliary_files[path]
    raw = client.complete(
        AUTHOR_SCRIPT_SYSTEM,
        REWRITE_SCRIPT_PROMPT.format(
            goal=DESCRIBE_GOAL,
            skill_name=skill,
            path=path,
            current_content=current,
            critique=critique or "(no critique provided)",
        ),
        tag="rewrite_script",
        max_tokens=2000,
    ).strip()
    raw = _JSON_FENCE.sub("", raw).strip()
    doc.auxiliary_files[path] = _ensure_shebang(path, raw) + (
        "\n" if not raw.endswith("\n") else ""
    )
    return out


def apply_remove_script(
    folder: SkillFolder,
    client: LLMClient,
    *,
    skill: str,
    path: str,
) -> SkillFolder:
    out = folder.clone()
    doc = out.by_name(skill)
    if not doc:
        raise KeyError(f"RemoveScript: skill '{skill}' not in folder")
    _validate_script_path(path)
    if path not in doc.auxiliary_files:
        raise KeyError(
            f"RemoveScript: '{path}' not in skill '{skill}'"
        )
    del doc.auxiliary_files[path]
    return out


def apply_rewrite_content(
    folder: SkillFolder,
    client: LLMClient,
    *,
    name: str,
    critique: str,
    skill_failures: str,
) -> SkillFolder:
    out = folder.clone()
    doc = out.by_name(name)
    if not doc:
        raise KeyError(f"RewriteSkillContent: '{name}' not in folder")
    body = client.complete(
        AUTHOR_B_SYSTEM,
        REWRITE_BODY_PROMPT.format(
            goal=DESCRIBE_GOAL,
            frontmatter=json.dumps(doc.frontmatter, indent=2),
            body=doc.body,
            critique=critique,
            skill_failures=skill_failures or "(no skill-specific failures attributed)",
        ),
        tag="rewrite_body",
    ).strip()
    doc.body = body
    return out


# ---------------------------------------------------------------------------
# Pick-op planner
# ---------------------------------------------------------------------------

def pick_op(
    folder: SkillFolder,
    critique: str,
    *,
    client: LLMClient,
    last_failures: Optional[List[Dict[str, Any]]] = None,
) -> OpRecord:
    """Ask the LLM for {op, args}; fall back to a rule on parse failure.

    The fallback rule: RewriteSkillContent on the lowest-success skill from
    ``last_failures``; if no failures attributed, rewrite the first skill.
    """
    raw = client.complete(
        OP_PLANNER_SYSTEM,
        OP_PLANNER_PROMPT.format(
            goal=DESCRIBE_GOAL,
            folder=folder.render_summary(body_chars=600),
            critique=critique,
        ),
        tag="op_planner",
        max_tokens=600,
    )
    parsed = _parse_json_object(raw)
    op = _coerce_op_record(parsed, folder)
    if op is not None:
        return op

    logger.warning("pick_op: LLM reply did not parse; falling back. raw=%r", raw[:200])
    return _fallback_op(folder, last_failures)


def _coerce_op_record(parsed: Optional[Dict[str, Any]], folder: SkillFolder) -> Optional[OpRecord]:
    if not isinstance(parsed, dict):
        return None
    op = parsed.get("op")
    if not isinstance(op, str):
        return None

    valid_names = set(folder.names())

    if op == "AddSkill":
        name = parsed.get("name")
        desc = parsed.get("description", "")
        seed = parsed.get("seed", "")
        if not isinstance(name, str) or not name or name in valid_names:
            return None
        return OpRecord(op, {"name": name, "description": desc, "seed": seed})

    if op == "RemoveSkill":
        name = parsed.get("name")
        if not isinstance(name, str) or name not in valid_names:
            return None
        if len(folder.skills) <= 1:
            return None  # would leave folder empty
        return OpRecord(op, {"name": name})

    if op == "RenameSkill":
        old = parsed.get("old")
        new = parsed.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        if old not in valid_names or new in valid_names or not new:
            return None
        return OpRecord(op, {"old": old, "new": new})

    if op == "SplitSkill":
        name = parsed.get("name")
        into = parsed.get("into")
        rationale = parsed.get("rationale", "")
        if not isinstance(name, str) or name not in valid_names:
            return None
        if not isinstance(into, list) or len(into) != 2:
            return None
        new_names = []
        for entry in into:
            if not isinstance(entry, dict):
                return None
            n = entry.get("name")
            if not isinstance(n, str) or not n:
                return None
            new_names.append(n)
        # Allow one new name to reuse the old slot (split + keep one child's name).
        free_after = (valid_names - {name})
        if any(n in free_after for n in new_names):
            return None
        if len(set(new_names)) != 2:
            return None
        return OpRecord(op, {"name": name, "into": into, "rationale": rationale})

    if op == "MergeSkills":
        a = parsed.get("a")
        b = parsed.get("b")
        into = parsed.get("into")
        desc = parsed.get("description", "")
        rationale = parsed.get("rationale", "")
        if not all(isinstance(x, str) and x for x in (a, b, into)):
            return None
        if a not in valid_names or b not in valid_names or a == b:
            return None
        free_after = valid_names - {a, b}
        if into in free_after:
            return None
        return OpRecord(op, {"a": a, "b": b, "into": into,
                             "description": desc, "rationale": rationale})

    if op == "RewriteSkillContent":
        name = parsed.get("name")
        excerpt = parsed.get("critique_excerpt", "")
        if not isinstance(name, str) or name not in valid_names:
            return None
        return OpRecord(op, {"name": name, "critique_excerpt": excerpt})

    if op == "AddScript":
        skill = parsed.get("skill")
        path = parsed.get("path")
        purpose = parsed.get("purpose", "")
        if not isinstance(skill, str) or skill not in valid_names:
            return None
        if not isinstance(path, str):
            return None
        try:
            _validate_script_path(path)
        except ValueError:
            return None
        # Collision check: AddScript must not target an existing path.
        doc = folder.by_name(skill)
        if doc is not None and path in doc.auxiliary_files:
            return None
        return OpRecord(op, {"skill": skill, "path": path, "purpose": purpose})

    if op == "RewriteScript":
        skill = parsed.get("skill")
        path = parsed.get("path")
        excerpt = parsed.get("critique_excerpt", "")
        if not isinstance(skill, str) or skill not in valid_names:
            return None
        if not isinstance(path, str):
            return None
        try:
            _validate_script_path(path)
        except ValueError:
            return None
        doc = folder.by_name(skill)
        if doc is None or path not in doc.auxiliary_files:
            return None
        return OpRecord(
            op, {"skill": skill, "path": path, "critique_excerpt": excerpt}
        )

    if op == "RemoveScript":
        skill = parsed.get("skill")
        path = parsed.get("path")
        if not isinstance(skill, str) or skill not in valid_names:
            return None
        if not isinstance(path, str):
            return None
        try:
            _validate_script_path(path)
        except ValueError:
            return None
        doc = folder.by_name(skill)
        if doc is None or path not in doc.auxiliary_files:
            return None
        return OpRecord(op, {"skill": skill, "path": path})

    return None


def _fallback_op(
    folder: SkillFolder,
    last_failures: Optional[List[Dict[str, Any]]],
) -> OpRecord:
    # Default: rewrite the skill whose invocation was associated with the
    # most failures, else the first skill in the folder.
    target = None
    if last_failures:
        counts: Dict[str, int] = {}
        for f in last_failures:
            for sk in f.get("skills_invoked") or []:
                counts[sk] = counts.get(sk, 0) + 1
        if counts:
            target = max(counts, key=lambda k: counts[k])
            if target not in folder.names():
                target = None
    if target is None and folder.skills:
        target = folder.skills[0].folder_name
    return OpRecord(
        "RewriteSkillContent",
        {"name": target, "critique_excerpt": "(fallback — LLM plan did not parse)"},
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_op(
    folder: SkillFolder,
    op: OpRecord,
    client: LLMClient,
    *,
    critique: str = "",
    skill_failures: str = "",
) -> SkillFolder:
    """Dispatch an OpRecord to the right apply_* function."""
    name = op.op
    a = op.args
    if name == "AddSkill":
        return apply_add_skill(
            folder, client,
            name=a["name"], description=a.get("description", ""),
            seed=a.get("seed", ""),
        )
    if name == "RemoveSkill":
        return apply_remove_skill(folder, client, name=a["name"])
    if name == "RenameSkill":
        return apply_rename_skill(folder, client, old=a["old"], new=a["new"])
    if name == "SplitSkill":
        return apply_split_skill(
            folder, client,
            name=a["name"], into=a["into"],
            rationale=a.get("rationale", ""),
        )
    if name == "MergeSkills":
        return apply_merge_skills(
            folder, client,
            a=a["a"], b=a["b"], into=a["into"],
            description=a.get("description", ""),
            rationale=a.get("rationale", ""),
        )
    if name == "RewriteSkillContent":
        return apply_rewrite_content(
            folder, client,
            name=a["name"],
            critique=critique or a.get("critique_excerpt", ""),
            skill_failures=skill_failures,
        )
    if name == "AddScript":
        return apply_add_script(
            folder, client,
            skill=a["skill"], path=a["path"],
            purpose=a.get("purpose", ""),
        )
    if name == "RewriteScript":
        return apply_rewrite_script(
            folder, client,
            skill=a["skill"], path=a["path"],
            critique=critique or a.get("critique_excerpt", ""),
        )
    if name == "RemoveScript":
        return apply_remove_script(
            folder, client,
            skill=a["skill"], path=a["path"],
        )
    raise ValueError(f"unknown op: {name}")
