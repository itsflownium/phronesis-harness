# Qwen3.5 2021 Dev Harness Result

Model: `Qwen/Qwen3.5-35B-A3B`

Serving setup: `vLLM`, `1x NVIDIA H200`, `FP8` weights, `BF16` KV cache.

Harness-selected score: `61/120`

Manual-audited estimate: about `62/120`

Manual uncertainty band: `60-64/120`

| Problem | Harness | Manual | Notes |
|---|---:|---:|---|
| A1 | 10 | 10 | Correct. |
| A2 | 10 | 10 | Correct. |
| A3 | 10 | 8 | Correct classification, but a fixable determinant/volume gap. |
| A4 | 2 | 1 | Wrong final answer and capped loop. |
| A5 | 10 | 10 | Correct. |
| A6 | 0 | 0 | Key contradiction not proved. |
| B1 | 0 | 0 | Token-looped. |
| B2 | 0 | 0 | Wrong answer and token-looped. |
| B3 | 10 | 10 | Correct. |
| B4 | 5 | 4 | Partial Wilson/Cassini route; prime case unfinished. |
| B5 | 2 | 8 | Harness under-scored a mostly correct DAG/triangularization proof. |
| B6 | 2 | 1 | Central inequality asserted, not proved. |
