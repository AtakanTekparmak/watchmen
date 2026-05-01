#!/usr/bin/env bash
# _prebuild_skillsbench.sh — pre-warm SkillsBench Docker images serially.
#
# Phase E launches many parallel inner trials, each requiring a built
# task image. If those builds happen concurrently they can deadlock on
# the Docker daemon under load. This script pre-builds every task
# image listed in <task-list-json> SERIALLY, with a 600s timeout per
# build, skipping images that already exist locally.
#
# In addition to the base task image, this script bakes the
# `claude-code` agent stack (Anthropic CLI v2.1.19 + Zed's claude-agent-acp
# ACP shim) into the resulting :prebuild tag. benchflow's per-trial
# install_cmd (registered by skill_evolve/agents/register_claude_code.py)
# starts with skip-guards:
#
#     ( command -v claude && [ "$(claude --version | awk '{print $1}')" = "2.1.19" ] || npm install ... ) &&
#     ( command -v claude-agent-acp || npm install ... )
#
# Pre-baking both binaries makes those guards fall through, saving the
# nodesource setup + npm-install cost on every one of Phase E's 1440
# trial containers (~60-90s each → ~24-36h cumulative wall-time at any
# concurrency).
#
# Usage:
#   bash scripts/_prebuild_skillsbench.sh <task-list-json>
#
# Default <task-list-json>: runs/skillsbench_baseline_v2/hot_12.json,
# falling back to skill_evolve/skillsbench/subset_17.json.

set -u  # NOT -e: a single failed build should not abort the whole walk.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_LIST="${1:-${REPO_ROOT}/runs/skillsbench_baseline_v2/hot_12.json}"
if [[ ! -f "${TASK_LIST}" ]]; then
  TASK_LIST="${REPO_ROOT}/skill_evolve/skillsbench/subset_17.json"
fi
if [[ ! -f "${TASK_LIST}" ]]; then
  echo "ERROR: no task list found; pass one explicitly" >&2
  exit 2
fi

VENDOR_ROOT="${REPO_ROOT}/skill_evolve/benchmark/vendor/skillsbench/tasks"
if [[ ! -d "${VENDOR_ROOT}" ]]; then
  echo "ERROR: vendor dir missing: ${VENDOR_ROOT}" >&2
  exit 2
fi

LOG_DIR="${REPO_ROOT}/runs/_prebuild_log"
mkdir -p "${LOG_DIR}"

# Pin matches skill_evolve/agents/register_claude_code.py:_CLAUDE_CODE_VERSION.
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.19}"

echo "[prebuild] task_list   = ${TASK_LIST}"
echo "[prebuild] vendor      = ${VENDOR_ROOT}"
echo "[prebuild] log_dir     = ${LOG_DIR}"
echo "[prebuild] claude-code = v${CLAUDE_CODE_VERSION}"

# Extract task IDs (strip optional "skillsbench/" prefix).
TASK_IDS=$(python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    for tid in json.load(f):
        print(tid.split("/", 1)[-1] if "/" in tid else tid)
' "${TASK_LIST}")

# Layer Dockerfile: takes a base ${BASE_IMAGE} arg, adds node 22 + claude-code +
# claude-agent-acp. Same install steps that benchflow's runtime install_cmd
# would otherwise execute on every trial container.
LAYER_DIR="$(mktemp -d -t prebuild_cclayer.XXXXXX)"
trap 'rm -rf "${LAYER_DIR}"' EXIT
cat > "${LAYER_DIR}/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG CLAUDE_CODE_VERSION=2.1.19
ENV DEBIAN_FRONTEND=noninteractive
# Node 22 + claude-code@<pin> + claude-agent-acp@latest, mirroring
# register_claude_code._INSTALL_CMD. We deliberately install for everyone:
# the host benchflow re-runs its own install_cmd inside, but its skip-guard
# will short-circuit (claude already present at the right version, and
# claude-agent-acp on PATH).
RUN set -eux; \
    if ! command -v node >/dev/null 2>&1 || \
       [ "$(node -e 'console.log(process.versions.node.split(\".\")[0])' 2>/dev/null || echo 0)" -lt 22 ]; then \
        apt-get update -qq && \
        apt-get install -y -qq curl ca-certificates && \
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
        apt-get install -y -qq nodejs && \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    npm install -g \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
        "@zed-industries/claude-agent-acp@latest"; \
    command -v claude; \
    claude --version; \
    command -v claude-agent-acp
DOCKERFILE

# probe_image_has_cc <image_tag> — returns 0 if image already has claude
# v${CLAUDE_CODE_VERSION} and claude-agent-acp on PATH; nonzero otherwise.
probe_image_has_cc() {
  local tag="$1"
  docker run --rm --entrypoint /bin/sh "${tag}" -c '
    set -e
    command -v claude-agent-acp >/dev/null 2>&1 || exit 1
    command -v claude >/dev/null 2>&1 || exit 1
    ver=$(claude --version 2>/dev/null | awk "{print \$1}")
    [ "$ver" = "'"${CLAUDE_CODE_VERSION}"'" ] || exit 1
  ' >/dev/null 2>&1
}

OK=0
SKIP=0
FAIL=0
LAYER_OK=0
LAYER_SKIP=0
LAYER_FAIL=0
for name in ${TASK_IDS}; do
  task_dir="${VENDOR_ROOT}/${name}"
  dockerfile="${task_dir}/environment/Dockerfile"
  image_tag="skillsbench/${name}:prebuild"
  if [[ ! -f "${dockerfile}" ]]; then
    echo "[prebuild] SKIP (no Dockerfile): ${name}" >&2
    ((SKIP+=1)) || true
    continue
  fi

  # Stage 1: build (or reuse) base task image.
  if docker image inspect "${image_tag}" >/dev/null 2>&1; then
    echo "[prebuild] cached base: ${image_tag}"
    ((SKIP+=1)) || true
  else
    log="${LOG_DIR}/${name}.log"
    echo "[prebuild] build base: ${name} -> ${image_tag} (log=${log})"
    if timeout 600 docker build -q -t "${image_tag}" \
        -f "${dockerfile}" "${task_dir}/environment" \
        >"${log}" 2>&1; then
      ((OK+=1)) || true
    else
      rc=$?
      echo "[prebuild] FAIL base (rc=${rc}): ${name} (see ${log})" >&2
      ((FAIL+=1)) || true
      continue  # no point baking a cc layer on a missing base
    fi
  fi

  # Stage 2: claude-code layer. Skip if probe shows it's already there.
  if probe_image_has_cc "${image_tag}"; then
    echo "[prebuild] cc layer present: ${image_tag}"
    ((LAYER_SKIP+=1)) || true
    continue
  fi
  cclog="${LOG_DIR}/${name}.cc.log"
  echo "[prebuild] bake cc layer: ${name} -> ${image_tag} (log=${cclog})"
  if timeout 600 docker build -q -t "${image_tag}" \
      --build-arg "BASE_IMAGE=${image_tag}" \
      --build-arg "CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}" \
      "${LAYER_DIR}" \
      >"${cclog}" 2>&1; then
    ((LAYER_OK+=1)) || true
  else
    rc=$?
    echo "[prebuild] FAIL cc layer (rc=${rc}): ${name} (see ${cclog})" >&2
    ((LAYER_FAIL+=1)) || true
  fi
done

# Final sanity sweep: confirm every freshly-tagged image actually has the
# stack benchflow's skip-guard expects.
echo "[prebuild] sanity check claude/claude-agent-acp on each :prebuild tag..."
SANITY_OK=0
SANITY_FAIL=0
for name in ${TASK_IDS}; do
  image_tag="skillsbench/${name}:prebuild"
  docker image inspect "${image_tag}" >/dev/null 2>&1 || continue
  if ver=$(docker run --rm --entrypoint /bin/sh "${image_tag}" -c \
        'claude --version 2>/dev/null | awk "{print \$1}"' 2>/dev/null) && \
     [ "${ver}" = "${CLAUDE_CODE_VERSION}" ] && \
     docker run --rm --entrypoint /bin/sh "${image_tag}" -c \
        'command -v claude-agent-acp' >/dev/null 2>&1; then
    echo "[prebuild]   OK   ${image_tag} -> claude ${ver} + claude-agent-acp"
    ((SANITY_OK+=1)) || true
  else
    echo "[prebuild]   FAIL ${image_tag} -> got ver='${ver:-<none>}'" >&2
    ((SANITY_FAIL+=1)) || true
  fi
done

echo "[prebuild] base:    ok=${OK} skip=${SKIP} fail=${FAIL}"
echo "[prebuild] cclayer: ok=${LAYER_OK} skip=${LAYER_SKIP} fail=${LAYER_FAIL}"
echo "[prebuild] sanity:  ok=${SANITY_OK} fail=${SANITY_FAIL}"
exit 0
