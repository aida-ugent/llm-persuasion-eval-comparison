#!/bin/bash
# Temperature-0 check: serve each local model and run bozdag2025 at temperature 0.0 on the first
# 20 items, replacing any existing __temp0 file (--fresh).
#
#     GPU=0 PORT=8000 setsid nohup bash scripts/temp0_rerun.sh > logs/temp0_rerun.log 2>&1 < /dev/null &
#     UTIL=0.5 on a shared GPU. llama-3.3-70b needs both GPUs: GPU=0,1.
set -u
cd ~/llm-persuasion-eval-comparison
if pgrep -f "core.runner --method bozdag2025" > /dev/null; then
  echo "ABORT: a bozdag2025 runner is already running"; exit 1
fi
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vllm4
PORT=${PORT:-8000}
export LOCAL_API_BASE_URL=http://localhost:$PORT/v1
ONLY=${MODELS:-}
TABLE=$(cat <<'EOF'
llama-3.2-3b     |meta-llama/Llama-3.2-3B-Instruct    |meta-llama/Llama-3.2-3B-Instruct    |
qwen2.5-7b       |Qwen/Qwen2.5-7B-Instruct            |Qwen/Qwen2.5-7B-Instruct            |
mistral-7b-v0.2  |mistralai/Mistral-7B-Instruct-v0.2  |mistralai/Mistral-7B-Instruct-v0.2  |
gemma-3-12b      |$HOME/models/gemma-3-12b-it         |google/gemma-3-12b-it               |
mistral-nemo-12b |mistralai/Mistral-Nemo-Instruct-2407|mistralai/Mistral-Nemo-Instruct-2407|
gemma-3-27b      |$HOME/models/gemma-3-27b-it         |google/gemma-3-27b-it               |
qwen3-32b        |Qwen/Qwen3-32B                      |Qwen/Qwen3-32B                      |--reasoning-parser qwen3
exaone-4.0.1-32b |LGAI-EXAONE/EXAONE-4.0.1-32B        |LGAI-EXAONE/EXAONE-4.0.1-32B        |--trust-remote-code
llama-3.3-70b    |meta-llama/Llama-3.3-70B-Instruct   |meta-llama/Llama-3.3-70B-Instruct   |--gpu-memory-utilization 0.95 --tensor-parallel-size 2
gpt-oss-120b     |openai/gpt-oss-120b                 |openai/gpt-oss-120b                 |--gpu-memory-utilization 0.95
EOF
)
serve() {
  pkill -f "vllm.entrypoints.*--port $PORT"; sleep 15
  local flags="$3"; local gpus=${GPU:-0}; [ "$4" = llama-3.3-70b ] && gpus=0,1
  [ -n "${UTIL:-}" ] && flags="$(echo "$flags" | sed 's/--gpu-memory-utilization [0-9.]*//') --gpu-memory-utilization $UTIL"
  CUDA_VISIBLE_DEVICES=$gpus HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} nohup python -m vllm.entrypoints.openai.api_server \
      --model "$1" --served-model-name "$2" --max-model-len 32768 --port $PORT $flags > "logs/vllm_temp0_$4.log" 2>&1 &
  for i in $(seq 1 180); do curl -s localhost:$PORT/v1/models | grep -q "$2" && return 0; sleep 10; done
  echo "SERVER FAILED for $2"; return 1
}
echo "$TABLE" | while IFS='|' read -r alias path served flags; do
  alias=$(echo $alias); path=$(eval echo $path | xargs); served=$(echo $served | xargs); flags=$(echo $flags | xargs)
  [ -z "$alias" ] && continue
  if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$alias,"; then continue; fi
  echo "################ $alias  $(date) ################"
  serve "$path" "$served" "$flags" "$alias" || continue
  python -u -m core.runner --method bozdag2025 --model $alias --set temperature=0.0 --tag temp0 --limit 20 --fresh 2>&1 | tail -3
done
pkill -f "vllm.entrypoints.*--port $PORT"
echo TEMP0DONE
