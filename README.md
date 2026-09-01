# Structure First

**Auditing Action Abstraction and Adaptive Feedback in ARC-AGI-3**

This repository is the reproducibility package for an ARC Prize 2026 Paper Track entry. It studies the action interface between an LLM agent's reasoning and its spatial commitments, using public human replays plus preregistered live-agent ablations.

## Final result

A heterogeneous structural frontier improves human-click-region coverage over a component-only control under the same 24-slot cap. Naive global adaptive reranking is not a universal improvement. A post-action state-change-rate gate improves cross-environment human-action alignment, but the preregistered five-game live test does **not** show a task-completion advantage: static, historical v47, and gated regime each clear `1/31` levels. The representative paper policy is therefore static.

## Repository map

- `paper/MANUSCRIPT.md` — final <=1,500-word Kaggle writeup source.
- `paper/CLAIM_LEDGER.md` — claim status, evidence, and limitations.
- `paper/REPRODUCIBILITY.md` — pinned replay revision, shard hashes, partitions, and byte-for-byte reproduction checks.
- `paper/LIVE_ABLATION_METHODS.md` — controlled live-agent methodology.
- `results/` — frozen replay and live-ablation JSON outputs.
- `analysis/` — replay coverage, reranking, LOEO-gate, and fixed-threshold evaluators.
- `policy/` — structural frontier implementation, overlay/build tooling, and tests.
- `notebook/arc_agi_3_action_interface_paper.ipynb` — public Paper Track companion notebook.
- `submission/arc_agi_3_representative_submission.ipynb` — full representative ARC-AGI-3 code-submission notebook.
- `assets/cover.png` — Paper Track Media Gallery cover.

## Evidence boundaries

Human-action retention is a proxy, not ARC task success. A frame changing after a MOUSE action is also not proof that the action caused the change or that objective progress occurred. The package preserves negative results rather than converting proxy gains into performance claims.

The pinned public replay mirror contains 340 sessions and 53,876 MOUSE decisions. ARC Prize separately announced 342 Public Demo plays; `paper/REPRODUCIBILITY.md` records this discrepancy without assuming an explanation.

## Open-source scope

Submitter-authored material is released under MIT-0 (`LICENSE`). Third-party material retains its upstream licensing and provenance; see `THIRD_PARTY_NOTICES.md` and `paper/LICENSE_AUDIT.md`.

The `RELEASE_MANIFEST.json` records the private research source commit and SHA256 of every exported file.
