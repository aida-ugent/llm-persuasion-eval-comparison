#!/bin/bash
# Collect the API arm: the five API models in parallel, each running the nine methods in series,
# then the 20% replicates. Needs the five API keys. Resumable.
#
#     nohup bash scripts/collect_api.sh > logs/collect_api.log 2>&1 &
cd "$(dirname "$0")/.."
METHODS="pauli2025 elaraby2024 bozdag2025 ju2025 liu2025 openai2024_makemepay openai2024_makemesay ahnert2025 amayuelas2024"
rep_n() {
  python - "$1" <<'PY'
import sys, yaml, math
n = yaml.safe_load(open("config/methods.yaml"))[sys.argv[1]]["n"]
print(max(4, math.ceil(0.2 * n)) if isinstance(n, int) else 4)
PY
}
one_model() {
  local a=$1
  for m in $METHODS; do
    echo "== $m ($a) $(date +%H:%M)"; python -u -m core.runner --method $m --model $a 2>&1 | tail -3
  done
  for m in $METHODS; do
    echo "== replicate $m ($a) $(date +%H:%M)"
    python -u -m core.runner --method $m --model $a --tag rep2 --limit $(rep_n $m) 2>&1 | tail -2
  done
  echo "MODELDONE $a $(date)"
}
mkdir -p logs
for a in gpt-5.6-sol claude-fable-5.1 deepseek-v4-flash kimi-k2.6 mistral-medium-3.5; do
  one_model $a > "logs/collect_api_$a.log" 2>&1 &
done
wait
echo ALLDONE
