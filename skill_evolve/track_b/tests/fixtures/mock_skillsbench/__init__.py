"""Mock SkillsBench fixture: 3 fake task IDs with deterministic pass/fail.

``mock_task_a`` and ``mock_task_b`` pass when the candidate bundle has a
SKILL.md (any one). ``mock_task_c`` always fails.

A deterministic backend stub lives in ``stub_backend.py``. Tests register
it via ``monkeypatch.setattr`` so the existing ``--agent-backend``
plumbing finds it.
"""
