#!/bin/bash
# Packaging check: rerun makemesay on llama-3.2-3b and qwen2.5-7b with the game-state notes folded
# into user turns, on the first 20 codewords, and print the means beside the original packaging.
# Writes only the tagged variant <model>__userfold.jsonl.
#
#     setsid nohup bash scripts/packaging_check.sh > logs/packaging_check.log 2>&1 < /dev/null &
#     tail -f logs/packaging_check.log
#
# Uses GPU=1 PORT=8001 by default. On a shared GPU set UTIL:
#     GPU=0 PORT=8000 UTIL=0.5 setsid nohup bash scripts/packaging_check.sh > logs/packaging_check.log 2>&1 < /dev/null &
set -u
cd ~/llm-persuasion-eval-comparison
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vllm4
PORT=${PORT:-8001}
export LOCAL_API_BASE_URL=http://localhost:$PORT/v1 CUDA_VISIBLE_DEVICES=${GPU:-1}
N=${N:-20}

serve() {  # hf id, log name
  pkill -f "vllm.entrypoints.*--port $PORT"; sleep 15
  HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} nohup python -m vllm.entrypoints.openai.api_server \
      --model "$1" --served-model-name "$1" --max-model-len 32768 --port $PORT \
      --gpu-memory-utilization ${UTIL:-0.9} > "logs/vllm_$2_userfold.log" 2>&1 &
  for i in $(seq 1 90); do curl -s localhost:$PORT/v1/models | grep -q "$1" && return 0; sleep 10; done
  echo "SERVER FAILED for $1"; return 1
}

for pair in "llama-3.2-3b|meta-llama/Llama-3.2-3B-Instruct" "qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct"; do
  alias=${pair%%|*}; hf=${pair#*|}
  echo "################ $alias  $(date) ################"
  serve "$hf" "$alias" || continue
  python -u -m core.runner --method openai2024_makemesay --model $alias \
      --set strict_alternation=true --tag userfold --limit $N 2>&1 | tail -3
done
pkill -f "vllm.entrypoints.*--port $PORT"

python - <<'PY'
import json
from pathlib import Path
def rows(p):  # skips a truncated line
    out = {}
    for l in open(p, encoding="utf-8"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if "item_id" in r:
            out[r["item_id"]] = r
    return out
print("\nmakemesay, first 20 codewords: original packaging vs folded")
print(f"{'model':14s} {'n':>3s} {'orig':>6s} {'fold':>6s} {'diff':>6s}  refused orig/fold")
for a in ("llama-3.2-3b", "qwen2.5-7b"):
    d = Path("data/results/openai2024_makemesay")
    o, f = rows(d / f"{a}.jsonl"), rows(d / f"{a}__userfold.jsonl")
    common = [k for k in f if k in o]
    sc = lambda R: [R[k]["score"] for k in common if R[k]["score"] is not None]
    mo, mf = (sum(sc(o)) / len(sc(o)) if sc(o) else float("nan")), (sum(sc(f)) / len(sc(f)) if sc(f) else float("nan"))
    ro, rf = sum(bool(o[k]["refused"]) for k in common), sum(bool(f[k]["refused"]) for k in common)
    print(f"{a:14s} {len(common):3d} {mo:6.3f} {mf:6.3f} {mf - mo:+6.3f}  {ro}/{rf}")
print("Read against the gate: makemesay's between-model spread was 0.70 on five models (n=20 each).")
PY
echo PACKDONE
