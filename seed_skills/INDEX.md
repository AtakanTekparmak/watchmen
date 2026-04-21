# Seed Skills Index

Baseline seed skills for the Hermes-agent skill evolution loop. Target benchmarks: terminal-bench (shell, debugging, file ops) and SWE-bench Verified (Python bug-fix patches). These are intentionally decent-but-not-perfect — the evolution loop is expected to mutate, specialize, merge, or drop them.

- **systematic-shell-debugging/** — Methodical response to failing shell commands: read error, inspect state, form one hypothesis, test minimally. Anti-pattern target: blind retries. (terminal-bench)
- **patch-then-verify/** — Bug-fix discipline: reproduce first, patch minimally, run the repro and the full test suite before declaring done. (SWE-bench Verified)
- **read-before-write/** — Never edit a file whose current contents you have not read this session; re-read the changed region after editing. (both benchmarks)
- **stop-and-replan/** — Anti-flailing: after 3 failed attempts on the same approach, halt, list what was tried, surface the shared assumption, pick a different angle. (both benchmarks)
- **ask-the-environment/** — Prefer cheap read-only inspection commands (`pwd`, `ls`, `which`, `git status`, `env`) over guessing about paths, versions, or state. (both benchmarks)
