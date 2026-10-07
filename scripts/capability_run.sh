#!/bin/bash
# Capability score for the fifteen models: MMLU-Pro on a seeded 300-item subset (5-shot CoT) and
# IFEval in full (prompt-level strict accuracy), through lm-eval-harness 0.4.9 over chat completions.
# Each local model also runs the second-persuadee subsample first (SKIP_P2=1 skips it).
# UTIL sets vLLM's GPU share on a shared GPU. Writes data/capability/<alias>/ and
# data/capability/summary.csv.
#
# One-time setup:
#     conda create -y -n capeval python=3.12 && conda activate capeval
#     pip install "lm_eval[api,ifeval]==0.4.9" datasets "transformers<5"
#     conda activate vllm4 && python scripts/capability_samples.py     # writes the 300-item sample
#
# Run:
#     SMOKE=1 bash scripts/capability_run.sh                    # llama-3.2-3b, 5 items, ~5 min
#     setsid nohup bash scripts/capability_run.sh > logs/capability.log 2>&1 < /dev/null &
#     ARM=api   bash scripts/capability_run.sh                  # the five API models only
#     MODELS=qwen3-32b,gpt-oss-120b bash scripts/capability_run.sh
set -u
cd ~/llm-persuasion-eval-comparison
source ~/miniconda3/etc/profile.d/conda.sh
# export the keys in .env without sourcing it
conda activate vllm4
eval "$(python -c 'from dotenv import dotenv_values; print("\n".join(f"export {k}={v!r}" for k, v in dotenv_values(".env").items() if v))')"
PORT=${PORT:-8001}; ARM=${ARM:-all}; ONLY=${MODELS:-}; SMOKE=${SMOKE:-0}
SAMPLES=data/capability/mmlu_pro_samples_lm_eval.json
[ -f "$SAMPLES" ] || { echo "run scripts/capability_samples.py first"; exit 1; }
SAMPLES_ARG="--samples $SAMPLES"; LIMIT=""
# lm-eval does not accept --limit together with --samples
[ "$SMOKE" = 1 ] && { LIMIT="--limit 5"; SAMPLES_ARG=""; ONLY=${ONLY:-llama-3.2-3b}; }
mkdir -p logs data/capability

# alias | path or hf id | served name | extra vLLM flags | reasoning
LOCAL=$(cat <<'EOF'
llama-3.2-3b     |meta-llama/Llama-3.2-3B-Instruct    |meta-llama/Llama-3.2-3B-Instruct    |                                                      |no
qwen2.5-7b       |Qwen/Qwen2.5-7B-Instruct            |Qwen/Qwen2.5-7B-Instruct            |                                                      |no
mistral-7b-v0.2  |mistralai/Mistral-7B-Instruct-v0.2  |mistralai/Mistral-7B-Instruct-v0.2  |                                                      |no
gemma-3-12b      |$HOME/models/gemma-3-12b-it         |google/gemma-3-12b-it               |                                                      |no
mistral-nemo-12b |mistralai/Mistral-Nemo-Instruct-2407|mistralai/Mistral-Nemo-Instruct-2407|                                                      |no
gemma-3-27b      |$HOME/models/gemma-3-27b-it         |google/gemma-3-27b-it               |                                                      |no
qwen3-32b        |Qwen/Qwen3-32B                      |Qwen/Qwen3-32B                      |--reasoning-parser qwen3                              |yes
exaone-4.0.1-32b |LGAI-EXAONE/EXAONE-4.0.1-32B        |LGAI-EXAONE/EXAONE-4.0.1-32B        |--trust-remote-code                                   |no
llama-3.3-70b    |meta-llama/Llama-3.3-70B-Instruct   |meta-llama/Llama-3.3-70B-Instruct   |--gpu-memory-utilization 0.95 --tensor-parallel-size 2|no
gpt-oss-120b     |openai/gpt-oss-120b                 |openai/gpt-oss-120b                 |--gpu-memory-utilization 0.95                         |yes
EOF
)
# alias | lm-eval model type | model id | base_url | key variable | reasoning
API=$(cat <<'EOF'
gpt-5.6-sol       |openai-chat-completions   |gpt-5.6-sol         |https://api.openai.com/v1/chat/completions   |OPENAI_API_KEY   |yes
claude-fable-5.1  |anthropic-chat-completions|claude-fable-5-1    |                                             |ANTHROPIC_API_KEY|yes
deepseek-v4-flash |openai-chat-completions   |deepseek-v4-flash   |https://api.deepseek.com/v1/chat/completions |DEEPSEEK_API_KEY |yes
kimi-k2.6         |openai-chat-completions   |kimi-k2.6           |https://api.moonshot.ai/v1/chat/completions  |MOONSHOT_API_KEY |yes
mistral-medium-3.5|openai-chat-completions   |mistral-medium-2604 |https://api.mistral.ai/v1/chat/completions   |MISTRAL_API_KEY  |no
EOF
)

want() { [ -z "$ONLY" ] || echo ",$ONLY," | grep -q ",$1,"; }
serve() {  # path, served, flags, alias
  conda activate vllm4
  pkill -f "vllm.entrypoints.*--port $PORT"; sleep 15
  local gpus=${GPU:-1}; [ "$4" = llama-3.3-70b ] && gpus=0,1
  local flags="$3"
  # UTIL replaces any utilization flag in the table
  [ -n "${UTIL:-}" ] && flags="$(echo "$flags" | sed 's/--gpu-memory-utilization [0-9.]*//') --gpu-memory-utilization $UTIL"
  CUDA_VISIBLE_DEVICES=$gpus HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} nohup python -m vllm.entrypoints.openai.api_server \
      --model "$1" --served-model-name "$2" --max-model-len 32768 --port $PORT $flags > "logs/vllm_cap_$4.log" 2>&1 &
  for i in $(seq 1 180); do curl -s localhost:$PORT/v1/models | grep -q "$2" && return 0; sleep 10; done
  echo "SERVER FAILED for $2"; return 1
}
persuadee2() {  # alias
  conda activate vllm4
  for m in bozdag2025 liu2025; do
    echo "== persuadee2 $m $1 $(date +%H:%M)"
    LOCAL_API_BASE_URL=http://localhost:$PORT/v1 python -u -m core.runner --method $m --model "$1" \
        --set persuadee=claude-haiku-4.5 --tag persuadee2 --limit 20 2>&1 | tail -2
  done
}
evaluate() {  # alias, model type, model_args, reasoning
  conda activate capeval
  local toks=2048; [ "$4" = yes ] && toks=8192
  echo "== lm_eval $1 $(date +%H:%M)"
  python scripts/lm_eval_run.py --model "$2" --model_args "$3" --tasks mmlu_pro,ifeval $SAMPLES_ARG $LIMIT \
      --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=$toks" \
      --seed 20261002 --output_path "data/capability/$1" --log_samples > "logs/lm_eval_$1.log" 2>&1
  grep "\[http\|^[A-Za-z_.]*Error" "logs/lm_eval_$1.log" | head -5; tail -12 "logs/lm_eval_$1.log"
}

if [ "$ARM" != api ]; then
  echo "$LOCAL" | while IFS='|' read -r alias path served flags reasoning; do
    alias=$(echo $alias); path=$(eval echo $path | xargs); served=$(echo $served | xargs); flags=$(echo $flags | xargs); reasoning=$(echo $reasoning)
    [ -z "$alias" ] && continue; want "$alias" || continue
    echo "################ $alias  $(date) ################"
    serve "$path" "$served" "$flags" "$alias" || continue
    [ "$SMOKE" = 1 ] || [ "${SKIP_P2:-0}" = 1 ] || persuadee2 "$alias"
    evaluate "$alias" local-chat-completions "model=$served,base_url=http://localhost:$PORT/v1/chat/completions,num_concurrent=16,max_retries=5,timeout=900" "$reasoning"
  done
  pkill -f "vllm.entrypoints.*--port $PORT"
fi
if [ "$ARM" != local ]; then
  echo "$API" | while IFS='|' read -r alias mtype mid url keyvar reasoning; do
    alias=$(echo $alias); mtype=$(echo $mtype); mid=$(echo $mid); url=$(echo $url); keyvar=$(echo $keyvar); reasoning=$(echo $reasoning)
    [ -z "$alias" ] && continue; want "$alias" || continue
    echo "################ $alias  $(date) ################"
    conc="num_concurrent=1,max_retries=30"
    if [ "$mtype" = anthropic-chat-completions ]; then
      evaluate "$alias" "$mtype" "model=$mid,$conc,timeout=900" "$reasoning"
    else   # lm-eval reads OPENAI_API_KEY for every OpenAI-compatible endpoint
      OPENAI_API_KEY=${!keyvar} evaluate "$alias" "$mtype" "model=$mid,base_url=$url,$conc,timeout=900" "$reasoning"
    fi
  done
fi

conda activate capeval
python - <<'PY'
import json, glob, csv
from pathlib import Path
rows = []
for d in sorted(Path("data/capability").iterdir()):
    if not d.is_dir(): continue
    files = sorted(d.rglob("results_*.json"))
    if not files: continue
    r = json.load(open(files[-1]))["results"]
    mp = r.get("mmlu_pro", {}).get("exact_match,custom-extract")
    ife = r.get("ifeval", {}).get("prompt_level_strict_acc,none")
    rows.append(dict(model=d.name, mmlu_pro_300=mp, ifeval_prompt_strict=ife, source=str(files[-1])))
    print(f"{d.name:20s} mmlu_pro {mp if mp is None else round(mp, 3)}   ifeval {ife if ife is None else round(ife, 3)}")
if rows:
    with open("data/capability/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> data/capability/summary.csv ({len(rows)} models)")
PY
echo CAPDONE
