# Paper release license audit

Verified 2026-08-31.

## Competition requirement

ARC Prize 2026 states that submitter-authored code and methods must be open sourced under a permissive public-domain-style license, giving CC0 and MIT-0 as examples. Third-party code must already permit public sharing under an open-source license.

Source: https://arcprize.org/competitions/2026

## Third-party lineage

- ARC Prize ARC-AGI toolkit and benchmarking code are MIT-licensed upstream.
- `sonpham-org/arc-3` documents its Duck Harness as a working fork of the Tufa Labs ARC-AGI-3 Duck Harness and explicitly states that the Tufa lineage is MIT-licensed.
- Preserve third-party copyright/license notices when redistributing copied or embedded code.

Sources:
- https://github.com/arcprize/ARC-AGI/blob/main/LICENSE
- https://github.com/sonpham-org/arc-3
## Release recommendation

Use MIT-0 for Patricia-authored paper code/methods unless the owner selects another eligible permissive license before publication. Scope that license explicitly to authored material and preserve upstream MIT provenance for third-party components. Do not change repository visibility until the final representative submission artifacts and release surface have been audited.
