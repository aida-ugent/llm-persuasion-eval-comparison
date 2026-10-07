#!/bin/bash
# Collect the local arm on the GPU server, one model at a time: serve it, run the nine methods,
# the 20% replicate of each cell and bozdag2025 at temperature 0, then unload. Resumable.
# Needs .env with OPENAI_API_KEY and the gemma overlays under ~/models.
#
#     setsid nohup bash scripts/collect_local.sh > logs/collect_local.log 2>&1 < /dev/null &
#     tail -f logs/collect_local.log
#
# Two lanes, each with its own GPU, port and model subset:
#     GPU=0 PORT=8000 MODELS=llama-3.2-3b,qwen2.5-7b,mistral-7b-v0.2,gemma-3-12b,mistral-nemo-12b \
#       setsid nohup bash scripts/collect_local.sh > logs/collect_lane0.log 2>&1 < /dev/null &
#     GPU=1 PORT=8001 MODELS=gemma-3-27b,qwen3-32b,exaone-4.0.1-32b,llama-3.3-70b,gpt-oss-120b \
#       setsid nohup bash scripts/collect_local.sh > logs/collect_lane1.log 2>&1 < /dev/null &
#
# llama-3.3-70b needs both GPUs; run it alone, last:
#     GPU=0,1 PORT=8000 MODELS=llama-3.3-70b \
#       setsid nohup bash scripts/collect_local.sh > logs/collect_llama70b.log 2>&1 < /dev/null &
set -u
cd ~/llm-persuasion-eval-comparison
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vllm4
PORT=${PORT:-8000}
export LOCAL_API_BASE_URL=http://localhost:$PORT/v1 CUDA_VISIBLE_DEVICES=${GPU:-0}
ONLY=${MODELS:-}                      # comma-separated aliases; empty = all ten
METHODS="pauli2025 elaraby2024 bozdag2025 ju2025 liu2025 openai2024_makemepay openai2024_makemesay ahnert2025 amayuelas2024"
# alias | model path or hf id | served name | extra vLLM flags
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

serve() {  # path, served name, flags, log name
  pkill -f "vllm.entrypoints.*--port $PORT"; sleep 15
  HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} nohup python -m vllm.entrypoints.openai.api_server \
      --model "$1" --served-model-name "$2" --max-model-len 32768 --port $PORT $3 > "logs/vllm_$4.log" 2>&1 &
  for i in $(seq 1 180); do curl -s localhost:$PORT/v1/models | grep -q "$2" && return 0; sleep 10; done
  echo "SERVER FAILED for $2"; return 1
}
rep_n() {  # 20% of the method's n, at least 4
  python - "$1" <<'PY'
import sys, yaml, math
n = yaml.safe_load(open("config/methods.yaml"))[sys.argv[1]]["n"]
print(max(4, math.ceil(0.2 * n)) if isinstance(n, int) else 4)
PY
}
run_model() {
  local a=$1
  for m in $METHODS; do
    echo "== $m ($a) $(date +%H:%M)"
    python -u -m core.runner --method $m --model $a 2>&1 | tail -3
  done
  for m in $METHODS; do
    echo "== replicate $m ($a) $(date +%H:%M)"
    python -u -m core.runner --method $m --model $a --tag rep2 --limit $(rep_n $m) 2>&1 | tail -2
  done
  echo "== temp0 bozdag2025 ($a) $(date +%H:%M)"
  python -u -m core.runner --method bozdag2025 --model $a --set temperature=0.0 --tag temp0 --limit 20 2>&1 | tail -2
}

echo "$TABLE" | while IFS='|' read -r alias path served flags; do
  alias=$(echo $alias); path=$(eval echo $path | xargs); served=$(echo $served | xargs); flags=$(echo $flags | xargs)
  [ -z "$alias" ] && continue
  if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$alias,"; then continue; fi
  echo "################ $alias  $(date) ################"
  serve "$path" "$served" "$flags" "$alias" && run_model "$alias"
done
pkill -f "vllm.entrypoints.*--port $PORT"
echo ALLDONE
