#!/usr/bin/env bash
# Clone every upstream method repository at the commit pinned in config/methods.yaml.
# Each clone is a shallow fetch of that one commit. Git LFS objects are not fetched.
#
#   bash scripts/setup_upstream.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export GIT_LFS_SKIP_SMUDGE=1

clone_at() {                     # clone_at <dir> <github owner/name> <full commit hash>
  local dir=$1 repo=$2 sha=$3
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]; then
    echo "ok       $dir already at ${sha:0:10}"; return
  fi
  echo "cloning  $repo @ ${sha:0:10} -> $dir"
  rm -rf "$dir"; mkdir -p "$dir"
  git init -q "$dir"
  git -C "$dir" remote add origin "https://github.com/$repo.git"
  if ! git -C "$dir" fetch -q --depth 1 origin "$sha"; then
    echo "         shallow fetch refused, fetching full history"
    git -C "$dir" fetch -q origin
  fi
  git -C "$dir" -c core.longpaths=true checkout -q --detach "$sha"
}

clone_at methods/elaraby2024/repo   EngSalem/Free-text-rationale-persuasion 96989c11fc4cfa3a6ccd1471c3dc64c0aa534ab5
clone_at methods/bozdag2025/repo    beyzabozdag/PersuadeMeIfYouCan          829f55eb22d0b9db0f29c82f72f95a750965e1c6
clone_at methods/ju2025/repo        KalinaEine/PsychologicalPersuasion      3c812f375d73c9f2fea597f30892cd6a6fa4cc70
clone_at methods/liu2025/repo       PLUM-Lab/PersuSafety                    892bf057f2974c1d1624f37d1110dcb4924e4fd6
clone_at methods/_shared/evals      openai/evals                            8eac7a7de5215c907fbddc30efdaf316913eccdd
# MakeMeSay reads its codeword lists from the clone, and openai/evals stores them in Git LFS
if git lfs version >/dev/null 2>&1; then
  git -C methods/_shared/evals lfs pull --include "evals/registry/data/make_me_say/*"
else
  echo "WARNING: git-lfs is not installed; install it and rerun, or MakeMeSay cannot load its codewords"
fi
clone_at methods/ahnert2025/repo    dess-mannheim/MeatlessAgents            5e2683b307ffcf71be4503f14968f99bd8b4ae9d
clone_at methods/amayuelas2024/repo amayuelas/multi-agent-attack            ce8ab36f7621914a51cb3b148368028a9979bd81

echo "done. Now: python -m core.registry --check"
