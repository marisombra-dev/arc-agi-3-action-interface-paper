# Paper Prize reproducibility record

## Human replay source

The analysis uses the open ARC-AGI-3 Public Demo human-testing dataset mirrored at Hugging Face as `magic-sword/arc_agi_3_public_demo_human_testing`.

Auto-converted Parquet revision: `447c0df44ec46b872ff5d148a705aacec449317a` (`refs/convert/parquet`, 2026-06-10).

The seven source shards are intentionally not committed to this Git repository. Reproduction should download them from the pinned dataset revision and verify these SHA256 digests before analysis:

| shard | bytes | SHA256 |
|---|---:|---|
| `0000.parquet` | 110727680 | `3c7560330407625fee2279c81e541449004ac495220f1334605f4bfaa7c07c82` |
| `0001.parquet` | 76502640 | `27fc4740b591a40d075cadc69fbfa14708d1d022c4ea60381ca83130a345baef` |
| `0002.parquet` | 14062017 | `4f04ca0ff73af47517163a64d612565a584f86054543e1db85fc08b6117a2c43` |
| `0003.parquet` | 95153406 | `d4d56a359f3ecea3a8b81a96aa70f306e2d8b623f2d64a416d664f78476bead8` |
| `0004.parquet` | 64456353 | `6c9827becdf523bef51c3ba170ba7a6ce50b0203d1055a89b1e50ea30fd1a347` |
| `0005.parquet` | 39275328 | `cabed0d906d00e4800beb72df45b0e56c4c688142a4249c2fca2f92f94b38de8` |
| `0006.parquet` | 74592860 | `0e298988a99d4a080ecf744dc518ad9a42d8350da0a4d28183572af0e191c218` |

## Frozen analysis partitions

Development data is shard `0000` only (50 sessions). Held-out causal evaluation uses shards `0001` through `0006` (290 sessions). The later leave-one-environment-out regime analysis combines the frozen development and held-out event tables but leaves each entire environment out when fitting its gate.

Raw replay census expected from all seven shards:

- sessions: `340`
- parsed trajectory events: `180496`
- MOUSE decisions: `53876`
- unique pre-click states: `42171`
- development-shard MOUSE decisions: `7526`
- development-shard unique pre-click states: `6090`

These counts are source-integrity checks, not fitted results.

## Analysis entry points

- `human_frontier_coverage.py`: static structural-frontier audit.
- `causal_frontier_rerank.py`: production-aligned v47 causal replay and frozen ablations.
- `regime_gate_loeo.py`: source-free leave-one-environment-out regime gate.
- `analyze_regime_activation.py`: descriptive activation analysis of the already-frozen gate.

Derived result JSONs under `paper_prize/results/` are committed. Intermediate event CSVs should be retained or regenerated from the pinned parquet source for independent verification.

## Static audit reproduction

On 2026-08-31 the seven pinned Parquet shards were restored from the public mirror and `human_frontier_coverage.py` was rerun from scratch with 8 workers.

The rerun reproduced exactly: 340 sessions, 53,876 MOUSE decisions, 42,171 unique pre-click states, and all committed static-frontier metrics.

The regenerated `human_frontier_summary.json` is byte-for-byte identical to the previously preserved result. SHA256 for both files:

`f4aab3655faa3308112f70f2d14ef9177025385a1313d82d5d0cbad8fb7c252f`

## Causal and regime-gate reproduction

On 2026-08-31 the production-aligned causal analysis was regenerated from the restored pinned Parquet shards using the frozen development/held-out partition, and the LOEO gate was then regenerated from those new intermediates plus the independently regenerated static event table. No replay-derived threshold or policy parameter was retuned during this reproduction.

Byte-for-byte reproduction checks:

- development causal summary SHA256: `fec4ac6f049a6e2738e388ff27fc8619f81d17eb4517c8cbddadd9dbbdfe4e68`
- held-out causal summary SHA256: `45248b22a211e02a6ebb2346cd285cf1c584bde298c1db4b770fa6bbc3f819ef`
- regenerated static event table SHA256: `9bc8314932c5abeae7cf90d2f6f8645d29185097e26d4116e9c30cc120ebdaf3`
- regenerated development causal event table SHA256: `cb4e97e5adbb4ee42e509c2a2a978e6ebac786456cf353a88973597468f1b8b1`
- regenerated held-out causal event table SHA256: `e0d7f4d83a7be8d045ff2f99628671970188a2e2824b218991e5e1bb2eb94c85`
- regenerated LOEO result SHA256: `ac27da88c3037dd0a259d94f80b4ee4de531cf0a3d5ad5b8d6f9bc3c31ffd4f6`

The regenerated held-out causal summary and LOEO JSON were each byte-for-byte identical to their committed historical counterparts. The rebuilt LOEO result retained all 18 folds, selected `effect_rate` in all 18 folds, and reproduced the committed macro gate improvement over static ranking (`+0.07954013479230164`). This closes the end-to-end replay chain from pinned public Parquets through raw-derived event tables to the frozen regime gate.

## Public replay count discrepancy

ARC Prize's 2026-04-14 human-dataset announcement describes the Public Demo set as `342` plays/replays. The pinned Hugging Face mirror used by this analysis exposes exactly `340` rows, and the seven pinned Parquet shards above reproduce that 340-session census.

This project therefore makes the narrower reproducible claim that its human-replay analyses cover **all 340 sessions present in the pinned mirror revision**, not necessarily every replay counted in ARC Prize's 342-play announcement. We have not found authoritative evidence establishing why two announced plays are absent from this mirror, so no explanation is assumed.

Relevant source records:
- ARC Prize, *Measuring Human Performance on ARC-AGI-3* (2026-04-14): 342 Public Demo plays/replays.
- `magic-sword/arc_agi_3_public_demo_human_testing`: 340-row public mirror used here.

## Matched-cap structural control

`matched_component_baseline.py` constructs the stronger component-only control under the same maximum 24-candidate budget as the heterogeneous frontier. On all 53,876 MOUSE decisions, rich top-24 region coverage is `48.752691%` versus `45.753211%` component-only (`+2.999480` points). Macro environment coverage is `62.5659%` versus `57.1175%` (`+5.4484` points). The committed result is `results/matched_component_baseline_24.json`.

This control supersedes the weaker legacy-subset comparison as the primary C1 baseline. The component control realizes fewer candidates on average, so the claim is extra human-region coverage under the same maximum cap, not greater information per candidate.

## Fixed deployed gate fidelity

`evaluate_fixed_regime_gate.py` applies the frozen runtime threshold `effect_rate >= 61/62` uniformly to the fully regenerated replay event tables. The committed result `results/fixed_regime_gate_61_62.json` has SHA256 `9f7976002d9724940945c3e73ba7e64c0ac0a616352dfc13b33eac6fd61ad8ef`.

The fixed rule reaches `69.299464%` macro top-8 retention versus `61.134748%` static (`+8.164716` points), improving 15/18 environments. This is a deployment-fidelity/sensitivity check on the same corpus, not a second independent held-out estimate, because `61/62` was summarized from the LOEO fold thresholds. The unbiased cross-environment estimate remains the LOEO learner result (`69.088762%`).
