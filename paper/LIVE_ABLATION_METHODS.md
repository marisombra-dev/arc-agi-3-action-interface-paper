# Live agent ablation methods freeze

Internal methods record. This consolidates settings already frozen before the regime result; it does not introduce a new experimental condition.

## r11l diagnostic

Three private Kaggle versions of the same notebook lineage target `r11l` only:

1. static structural frontier;
2. historical v47 causal/novelty ranking;
3. frozen state-change-rate regime gate (`0.9838709677419355`).

Common execution controls:
- model: `Qwen3.6-27B-AWQ` (`bachhg/qwen3-6-27b-awq/Transformers/awq/1`);
- Kaggle machine: Nvidia Tesla T4, tensor parallel size 2;
- internet disabled;
- analyzer context window: 8,192 tokens;
- analyzer temperature: `0.6`;
- top-p: `0.95`;
- top-k: `20`;
- no `LOCAL_ANALYZER_SEED` override, so live decoding is stochastic;
- post-setup soft runtime: 3,600 seconds;
- action-only MOUSE decisions require a 1-based `candidate_index` from the visible 16-item HOST_INTERACTION menu.

## Policy isolation

The v48 five-game and r11l notebook archives were audited pairwise. Each embedded archive contains 78 files, and each policy pair differs only in `inference/agent/tool_agent.py`.

The regime arm replaces the v47 ranking call with `rank_interventions_regime_gated`. When prior rolling MOUSE state-change rate is below the frozen threshold, it uses the static frontier. At or above threshold, it uses outcome-only feedback.

The gate feature is temporal state change after MOUSE, not proof that the action caused the change. This semantic limitation was frozen before the live regime result.

## Conditional five-game validation

If and only if the preregistered Gate-3 activation rule fires, the broader validation uses `r11l`, `sp80`, `m0r0`, `ft09`, and `ls20` under the same model/hardware/runtime lineage.

To reduce avoidable sampling variance, all three broader arms set the same predeclared `LOCAL_ANALYZER_SEED=20260831` while retaining temperature `0.6`, top-p `0.95`, and top-k `20`.

This common seed is a design improvement for the conditional broader experiment. It must not be retroactively treated as if it had controlled the already-run unseeded r11l diagnostic.
