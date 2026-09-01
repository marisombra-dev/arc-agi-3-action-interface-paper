# Structure First: Auditing Action Abstraction and Adaptive Feedback in ARC-AGI-3

## Abstract

ARC-AGI-3 turns reasoning into an interactive control problem: an agent must infer goals and act without instructions, while coordinate-based actions expose thousands of possible commitments. We study the action interface itself. A source-free structural frontier compresses a 64x64 MOUSE surface into at most 24 indexed candidates derived from visual components, motifs, islands, gaps, and fallback probes. Across 53,876 human MOUSE decisions in 18 public environments, the frontier retains 48.753% of human-selected regions, versus 45.753% for a component-only control under the same 24-slot cap. We then test whether transition feedback should rerank this prior. Historical causal/novelty reranking and an outcome-only policy both fail as universal improvements. Environment-held-out analysis instead selects prior post-action state-change rate as a gate for feedback override. Preregistered live ablations separate this human-action proxy from actual ARC performance. In preregistered common-seed five-game validation, the regime clears 1/31 levels versus 1/31 static and 1/31 historical-v47, so replay alignment does not translate into a gate-specific task-completion win.

## Introduction

ARC-AGI-3 asks an agent to explore unfamiliar interactive worlds, infer their objectives, and act efficiently [1]. That creates a difficulty that precedes high-level planning: what should the model be allowed to choose as an action?

For MOUSE actions on a 64x64 grid, unconstrained coordinate emission exposes 4,096 possible positions. A language model can reason well about a scene yet still waste interactions because its final spatial commitment is poorly grounded. Our starting hypothesis is therefore modest: improve the interface between reasoning and action before attributing every failure to reasoning itself.

We replace raw coordinate choice with a compact structural frontier. The model sees indexed candidate regions extracted from the current frame and commits by candidate ID, while retaining a raw-coordinate escape path. The main question is not whether structured grounding exists; it clearly does in prior GUI and ARC work. Our question is empirical: how much useful action space does a source-free frontier preserve across heterogeneous environments, and when should observed transition feedback be allowed to override that structural prior?

## Prior work and contribution boundary

Indexed or region-based visual grounding is established: Set-of-Mark labels image regions and GUI-Actor proposes coordinate-free candidate regions [2,3]. ARC-AGI-3-specific graph explorers already derive click targets from connected components and track tested state-action pairs [4]. Human replays have likewise been used publicly for behavioral cloning and action priors [5].

We therefore do not claim any of those primitives as new. The contribution is the empirical package: a heterogeneous source-free frontier, a controlled human-replay audit against a same-cap component baseline, explicit falsification of naive adaptive reranking, environment-held-out selection of a feedback-override condition, and live tests that keep proxy alignment separate from task performance.

## Approach

For each frame, the structural frontier extracts connected components and augments them with higher-order spatial hypotheses including multicolor islands, bounded motifs, enclosed regions, lattice gaps, and a small fallback probe set. Candidates are deduplicated by position, ranked structurally, and capped at 24. The runtime action schema exposes a smaller visible menu and requires a 1-based candidate index for MOUSE commitment when candidates exist. Raw coordinates remain available as an escape hatch.

We evaluated candidate coverage on the public human-replay corpus rather than training the agent to imitate particular games. The pinned mirror contains 340 replay sessions and 53,876 MOUSE decisions across 18 MOUSE environments. For each pre-click frame, we ask whether the human click falls inside one of the frontier's top-24 candidate regions. A matched maximum-budget control uses the existing component ranking but is also capped at 24 candidates.

We next replayed interaction histories through three ranking policies. Static preserves the structural order. Historical v47 adds novelty and transition-effect pressure. Outcome-only uses observed post-action outcomes without the historical novelty machinery. Because neither adaptive policy was universally superior, we fit a one-feature threshold stump in leave-one-environment-out evaluation. Each fold chooses its feature and threshold using the other 17 environments only. All 18 folds select prior MOUSE post-action state-change rate. The median selected threshold, 61/62, was frozen before live tests and used as the deployed regime gate: below it, retain static ordering; at or above it, permit outcome-only feedback to rerank candidates.

State-change rate is deliberately descriptive. A changed frame after a click is not proof that the click caused the change, and it is not equivalent to objective progress.

For reproducibility, the seven replay shards are revision-pinned and hash-pinned; the static, causal, held-out, and LOEO outputs were regenerated from raw-derived inputs and matched the preserved results byte-for-byte.

## Results

The rich top-24 frontier covers 48.753% of human click regions event-weighted. The same-cap component-only control covers 45.753%, a +2.999 percentage-point gain. Across environments, macro coverage is 62.566% versus 57.117% (+5.448 points): the rich frontier improves 13 environments, ties one, and hurts four. This is extra coverage under a fixed maximum budget, not evidence that every rich candidate is more informative; the component control realizes fewer candidates on average (21.005 versus 23.394).

A separate systems failure also mattered: early runs paired solver concurrency 16 with a one-sequence vLLM server and collapsed into timeouts; serial serving restored complete evaluation.

Adaptive reranking produced the opposite of the easy success story. On the development replay, historical v47 reduces top-8 retention among represented human targets from 43.084% static to 30.742%. On frozen held-out shards, event-weighted outcome-only feedback reaches 67.834% versus 77.944% static. Environment-macro aggregation later makes outcome-only nearly tie static, revealing strong heterogeneity rather than a universal benefit.

The environment-held-out gate is different. Fold-specific LOEO rules reach 69.089% macro top-8 retention versus 61.135% static and 61.339% outcome-only-everywhere, improving 14 of 18 held-out environments. The selected feature is state-change rate in every fold. Applying the frozen single threshold `61/62` uniformly to the full replay corpus gives 69.299% macro retention (+8.165 points versus static), within 0.211 points of the fold-specific learner. This is a deployment-fidelity check rather than a second held-out estimate.

In an unseeded r11l diagnostic using Qwen3.6-27B-AWQ on a T4, static, historical v47, and the frozen regime all clear 0/6 levels. The regime nevertheless activates as designed and changes the coordinate attached to the model's chosen candidate index on 27 of 30 gate-active decisions. This establishes behavioral participation, not improvement.

The preregistered broader test uses the same five games and common seed 20260831 for static, v47, and regime; the seed reduces avoidable sampling variance but cannot create matched counterfactuals after prompts diverge. All arms share the same serving and anti-stall compatibility layer. Static clears 1/31 total levels, entirely on ls20. Historical v47 clears 1/31 total levels. Regime clears 1/31 versus static 1/31 and historical v47 1/31. Because it does not strictly exceed both baselines, the frozen primary rule does not support C5 in this five-game systems sample.

## Limitations and conclusion

A historical v47 ARC-AGI-3 submission scores 0.09; the final paper-linked submission will use the policy selected by the frozen live comparison. Either way, this work is not evidence of a strong ARC solver. Human-action retention is also only a proxy: people can click inefficiently, and reproducing human-used regions does not imply discovering goals or planning correctly. Coverage varies sharply by environment, and the five-game live ablation is a small controlled systems sample rather than a statistical estimate of general agent performance.

The useful result is narrower. A model's reasoning is mediated by its action interface. Heterogeneous structural proposals can preserve more human-used spatial commitments than a component-only frontier at the same cap, but adding adaptive feedback indiscriminately can destroy that useful prior. In these replays, feedback becomes safer to expose only in a near-certain post-action state-change regime. For interactive reasoning systems, our practical recommendation is therefore: compress commitment, measure what the interface preserves, ablate feedback before trusting it, and separate changes in proxy alignment from changes in task success.

## References

[1] ARC Prize Foundation. *ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence*. arXiv:2603.24621, 2026.  
[2] Yang et al. *Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V*. arXiv:2310.11441, 2023.  
[3] Wu et al. *GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents*. NeurIPS, 2025.  
[4] Rudakov, Shock & Cowley. *Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks*. arXiv:2512.24156, 2025.  
[5] AR6420. *ARC-AGI-3 Agent: Modular Behavioral Cloning and Frame-Change Auxiliary Agent*. Public repository, 2026.
