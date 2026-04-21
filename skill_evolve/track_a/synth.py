"""A/B synthesizer — combines two candidate folders into a third.

Prompts the LLM for a JSON-shaped folder (see SYNTH_PROMPT) and rebuilds
a SkillFolder from it. Returns ``None`` if the response can't be parsed
into a valid folder — the caller (runner) then drops AB from the
tournament for that pass.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .folder import SkillDoc, SkillFolder, make_skill_doc
from .llm import LLMClient
from .prompts import DESCRIBE_GOAL, SYNTH_PROMPT, SYNTH_SYSTEM
from .validate import validates

logger = logging.getLogger(__name__)


def synthesize(
    a: SkillFolder,
    b: SkillFolder,
    client: LLMClient,
    *,
    rng: Optional[random.Random] = None,
) -> Optional[SkillFolder]:
    """Ask the LLM to merge A and B. Returns a valid SkillFolder or None."""
    rng = rng or random.Random()
    # Shuffle order so the synthesizer doesn't systematically prefer one input.
    if rng.random() < 0.5:
        x, y = a, b
    else:
        x, y = b, a

    raw = client.complete(
        SYNTH_SYSTEM,
        SYNTH_PROMPT.format(
            goal=DESCRIBE_GOAL,
            x=x.render_summary(body_chars=1500),
            y=y.render_summary(body_chars=1500),
        ),
        tag="synth",
        max_tokens=8192,
    )
    folder = _parse_synth_response(raw)
    if folder is None:
        logger.warning("synthesize: could not parse response as JSON folder")
        return None
    if not validates(folder):
        logger.warning("synthesize: parsed folder did not validate; returning None")
        return None
    return folder


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_synth_response(text: str) -> Optional[SkillFolder]:
    s = _JSON_FENCE.sub("", text).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    skills = obj.get("skills")
    if not isinstance(skills, list):
        return None
    docs: List[SkillDoc] = []
    for entry in skills:
        if not isinstance(entry, dict):
            return None
        folder_name = entry.get("folder_name") or entry.get("name")
        if not isinstance(folder_name, str) or not folder_name:
            return None
        description = entry.get("description", "") or ""
        body = entry.get("body", "") or ""
        if not isinstance(description, str) or not isinstance(body, str):
            return None
        docs.append(make_skill_doc(
            folder_name,
            name=entry.get("name") or folder_name,
            description=description,
            body=body,
        ))
    if not docs:
        return None
    return SkillFolder(skills=docs)
