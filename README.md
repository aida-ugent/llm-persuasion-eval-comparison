# LLM Persuasiveness Evaluation Method Comparison

Nine published methods for evaluating how persuasive a language model is, run through one harness
on the same fifteen models (135 runs). We try to answer the question: do the methods agree on how they rank the
models, and if not, why?

Paper: *LLM Persuasiveness Is in the Eye of the Evaluation* (Dementaviciute et al., 2026).

## Methods

| method_id | Name in the paper | Paper | Code it follows |
|---|---|---|---|
| pauli2025 | Rewrite Text | Pauli et al. 2025, arXiv 2406.17753 | `APauli/Persuasive_language_in_pairs` (HuggingFace) |
| elaraby2024 | Rationale | Elaraby et al. 2024, arXiv 2406.13905 | EngSalem/Free-text-rationale-persuasion |
| bozdag2025 | Subjective Opinion | Bozdag et al. 2025, arXiv 2503.01829 | beyzabozdag/PersuadeMeIfYouCan |
| ju2025 | False Claim | Ju et al. 2025, arXiv 2506.06800 | KalinaEine/PsychologicalPersuasion |
| liu2025 | Benign Request | Liu et al. 2025, arXiv 2504.10430 | PLUM-Lab/PersuSafety |
| openai2024_makemepay | MakeMePay | OpenAI 2024, arXiv 2412.16720 | openai/evals |
| openai2024_makemesay | MakeMeSay | OpenAI 2024, arXiv 2412.16720 | openai/evals |
| ahnert2025 | Reduce Meat | Ahnert et al. 2025, arXiv 2504.04872 | dess-mannheim/MeatlessAgents |
| amayuelas2024 | Mislead Debate | Amayuelas et al. 2024, arXiv 2406.14711 | amayuelas/multi-agent-attack |

All settings are in `config/`: `methods.yaml` (one block per method, with every deviation from
the original listed under `deviation:`; [DEVIATIONS.md](DEVIATIONS.md) is the same list as one table), `models.yaml` (the fifteen models) and `pinned.yaml`
(what is held constant: persuadee gpt-4.1-mini, judge gpt-4o, seed, refusal policy).
Model aliases follow the provider ids; `deepseek-v4-flash` is DeepSeek-V4.1-Flash in the paper, served
by the provider as `deepseek-flash` (see `data/results/_model_lock.json`).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env                 # add your API keys
bash scripts/setup_upstream.sh       # clones the original repos at the commits we used
python -m core.registry --check      # should end with "config OK"
python tests/test_prompts.py         # prompts match the clones
```

The adapters read their prompts from the original repos, so the clones are required.

## Running

```bash
python -m core.runner --method liu2025 --model gpt-5.6-sol --test   # 2 items, writes nothing
python -m core.runner --method liu2025 --model gpt-5.6-sol          # full run, resumable
python -m core.schedule                                             # all 135 runs, with the vLLM command per local model
```

Results go to `data/results/<method>/<model>.jsonl`, one row per item, with the run settings in
the first line. Local models need a vLLM server; the persuadee and judge are OpenAI models, so
every run needs an OpenAI key.

## What is here

```
core/        config loading, API clients, runner, scheduler
config/      methods, models, pinned agents, pricing
methods/     one adapter per method (the original repo is cloned next to it)
analysis/    build_matrix.py turns the raw results into the matrix and statistics
scripts/     collection drivers, sensitivity checks, repairs
data/        results/    raw rows of every run
             derived/    the numbers reported in the paper (matrix.csv, rq1.json, ...)
             capability/ MMLU-Pro scores used as the capability covariate
```

`python analysis/build_matrix.py` rebuilds the matrix and the rq1/rq2 files from `data/results/`; the other derived files come from the scripts named in their docstrings.

## Licence

Code: MIT (see LICENSE). Data under `data/`: CC BY 4.0. The files under `methods/*/data/` are
copies from the original repositories and keep their own licences.
