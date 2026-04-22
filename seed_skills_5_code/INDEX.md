# Seed Skills Index (E3 — 5 code-bearing skills)

Five upstream-sourced SWE skills covering diverse axes, each with the
scripts/references that ship upstream. Seeds the "code allowed, rich
seed corpus" arm of the skills-with-code ablation. Source axes span
bug-hunting, pre-ship verification, cybersecurity shell+JSON, CLI /
evaluation scaffolding, and canonical-output webapp testing.

- **systematic-debugging/** — 4-phase root-cause investigation + `find-polluter.sh` bisection helper. Source: [obra/superpowers](https://github.com/obra/superpowers/tree/main/skills/systematic-debugging) (MIT).
- **verification-before-completion/** — Gate-function discipline: run verification before claiming success. Prose-only upstream. Source: [obra/superpowers](https://github.com/obra/superpowers/tree/main/skills/verification-before-completion) (MIT).
- **analyzing-persistence-mechanisms-in-linux/** — Linux persistence vector scan (crontab, systemd, LD_PRELOAD, bashrc, authorized_keys) + `agent.py` helper + API reference. Source: [mukul975/Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills/tree/main/skills/analyzing-persistence-mechanisms-in-linux) (Apache-2.0).
- **mcp-builder/** — MCP server authoring guide + `connections.py`, `evaluation.py` runners, best-practices reference. Source: [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/mcp-builder) (Apache-2.0).
- **webapp-testing/** — Playwright-based local webapp test toolkit + `with_server.py` lifecycle helper + 3 example scripts. Source: [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/webapp-testing) (Apache-2.0).
