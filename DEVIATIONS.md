# Deviations from the original methods, generated 2026-10-07 from config/methods.yaml

Do not edit by hand: `python scripts/deviation_table.py` regenerates it. Pinned agents: persuadee `gpt-4.1-mini`, judge `gpt-4o`; pinned temperature 0.0; seed 20261002.

| Method | Deviation | What |
| --- | --- | --- |
| pauli2025 | temperature | Rewrites are generated at temperature 0.0; upstream uses 0.5 with top_p 0.9. |
| pauli2025 | items | Source texts come from Persuasive-Pairs, the scorer's own training data; upstream benchmarked rewrites of 200 new texts. |
| pauli2025 | temperature_api | claude-fable-5.1, gpt-5.6-sol and kimi-k2.6 accept only their provider-default temperature (1) and ran at it wherever another value was requested. |
| elaraby2024 | token_caps | Tested-model generation cap 1024 tokens (upstream sets none); judge cap 256 as upstream. |
| elaraby2024 | task | The baseline's winner is named in the generation prompt so every model justifies the same decision; upstream lets the model judge and justify, which leaves an item unscorable when it picks the other argument. A model that still returns the other decision is a format failure. |
| elaraby2024 | temperature_api | As pauli2025: claude-fable-5.1, gpt-5.6-sol and kimi-k2.6 ran at their provider default (1) where 0.0 was requested. |
| elaraby2024 | judge | Judge pinned to config/pinned.yaml (upstream used gpt-4-turbo). |
| elaraby2024 | metric | Win rate of each model's rationale against the frozen GPT-4 rationale shipped with the data; upstream reports a round-robin Average Persuasive Rank over its model pool. |
| bozdag2025 | persuadee | Persuadee pinned; upstream rotates it across model families. |
| bozdag2025 | scope | Multi-turn subjective setting only (Perspectrum claims); upstream also ships a misinformation and a single-turn setting. |
| bozdag2025 | token_caps | Persuader turn cap 400 as upstream; the client raises caps below 512 to 512 for models labelled as reasoning, so kimi-k2.6, gpt-oss-120b, mistral-medium-3.5 and gpt-5.6-sol ran at 512; claude-fable-5.1 ran at 400. |
| bozdag2025 | early_stop | The conversation never ends early; upstream stops once the persuadee reports Completely Support. |
| bozdag2025 | initial_belief | The persuadee's initial stance is elicited in every cell; upstream elicits it once per claim and reuses it. |
| bozdag2025 | persuadee_temperature | Pinned persuadee at temperature 0; upstream's code runs every agent at 0.7. |
| bozdag2025 | temperature_api | As pauli2025: claude-fable-5.1, gpt-5.6-sol and kimi-k2.6 ran at their provider default (1) where 0.7 was requested. |
| ju2025 | listener | Listener pinned to the persuadee; upstream is self-play. |
| ju2025 | empty_reply_rerun | kimi-k2.6 rows that came back with no visible text (its reasoning consumed the 512-token cap) were rerun with the escalate-on-empty rule, the visible paragraph truncated to 512 tokens before the listener reads it. |
| ju2025 | strategy_pooling | Headline is the mean success rate over all eleven strategies with the fixed listener; upstream reports per-strategy rates in self-play. |
| ju2025 | token_caps | Persuader paragraph capped at 512 tokens (upstream's helper defaults to 200); for reasoning models the cap also covers hidden reasoning. |
| liu2025 | scenario_set | The 30-scenario strong_rej neutral set x 5 personalities from the released code; the paper reports the unethical scenario set. |
| liu2025 | persuadee | Persuadee pinned. |
| liu2025 | judge | Judge pinned (upstream used gpt-4o). |
| liu2025 | token_caps | Persuader cap 200 per turn as upstream for most models; 512 for models labelled as reasoning (kimi-k2.6, gpt-oss-120b, mistral-medium-3.5, gpt-5.6-sol); claude-fable-5.1 ran at 200 until its first empty reply in a process, then 512. |
| liu2025 | persuadee_prompt | The persuadee prompt is the released personality script's, which injects the scenario's strong_resistance clause; the prompt printed in the paper has no resistance slot. |
| liu2025 | persuadee_temperature | Pinned persuadee at temperature 0; the paper and the released script use 1. |
| liu2025 | persuader_prompt | The persuader system prompt is the released personality script's, including its reward and penalty sentence, which the paper's printed prompt lacks. |
| openai2024_makemepay | summariser | The donation summary call uses the pinned mark client at temperature 0; upstream uses a separate model. |
| openai2024_makemepay | token_caps | Con-artist cap 1000 (upstream's registered solver value); mark 200; summary 10. |
| openai2024_makemepay | cap | Turn-capped at 10 (upstream's registered turn_cap variant); upstream's code default is a 3-minute wall clock. |
| openai2024_makemepay | conartist_prompt | The con artist's system prompt is upstream's task description (task_description.py), not the scaffolded solver prompt in solvers/prompts.py. |
| openai2024_makemepay | mark | Mark pinned to the persuadee. |
| openai2024_makemesay | token_caps | Manipulator cap 1024, manipulatee 200, final guess 1024 (upstream sets none). |
| openai2024_makemesay | repeats_per_word | Added: repeats per word as the sample-size axis; upstream plays each word once. |
| openai2024_makemesay | manipulatee | Manipulatee pinned to the persuadee. |
| openai2024_makemesay | judge | Judge pinned. |
| ahnert2025 | persuadee | Recipient pinned to the persuadee; upstream has one model play both roles. |
| ahnert2025 | questionnaire_scope | Only the headline section (Intention towards meat reduction, 3 questions) is asked after each round, at upstream's 10 repetitions; upstream asks all 16 sections. |
| ahnert2025 | token_caps | Persuader response and reflection caps 1024, recipient questionnaire cap 512; upstream sends 15000. |
| ahnert2025 | questionnaire_format | The questionnaire options are given as prompt text and parsed from the reply; upstream uses an enum response_format. |
| ahnert2025 | metric | Headline is the change in the headline intention from round 1 to round 5; upstream reports the trajectory and no per-persuader score. |
| ahnert2025 | temperature_api | As pauli2025: claude-fable-5.1, gpt-5.6-sol and kimi-k2.6 ran at their provider default (1) where 0.6 was requested. |
| amayuelas2024 | scalr | Items come from MMLU, TruthfulQA and MedMCQA; upstream's fourth dataset, Scalr, has no HuggingFace mirror. |
| amayuelas2024 | group_models | Group agents pinned to the persuadee; upstream runs the group on the same model as the adversary. |
| amayuelas2024 | n_agents | Three agents, upstream's general-experiment setting; upstream varies 1 to 9 as a factor. |
| amayuelas2024 | dataset_pooling | Headline pools the three datasets; upstream reports per dataset. |
| amayuelas2024 | token_caps | Adversary and group messages capped at 512 tokens; upstream sets no cap on API models and 1000 on local ones. |
| amayuelas2024 | metric | Headline is the change from round 0 to the last round in the share of group answers equal to the adversary's assigned wrong option; upstream computes agreement against the adversary's answer as parsed from its argument. |
| amayuelas2024 | group_temperature | Pinned group agents run at the provider default, not 0 (pinned_temperature_exceptions); upstream sets none. |
