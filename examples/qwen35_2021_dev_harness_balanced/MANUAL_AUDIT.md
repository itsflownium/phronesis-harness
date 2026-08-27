# Qwen3.5 2021 Dev Harness Result

Model: `Qwen/Qwen3.5-35B-A3B`

Serving setup: `vLLM`, `1x NVIDIA H200`, `FP8` weights, `BF16` KV cache.

Harness-selected score: `61/120`

Manual official-style score: **`62/120`**

Manual uncertainty band: `60-64/120`

This is an unofficial point estimate from a proof-by-proof comparison with the published 2021 answers and solutions. The `60-64/120` range reflects normal subjectivity in Putnam partial credit.

| Problem | Harness | Manual | Audit finding |
|---|---:|---:|---|
| A1 | 10 | 10 | Correct lower bound and valid 578-hop construction. |
| A2 | 10 | 10 | Correct logarithmic derivative and limit. |
| A3 | 10 | 8 | Correct classification, but the tetrahedron-volume determinant identity is off by a factor of four. |
| A4 | 2 | 1 | Notices leading asymptotic cancellation but gives the wrong final value. |
| A5 | 10 | 10 | Correct residue counts and finite-field power-sum argument. |
| A6 | 0 | 0 | Correct conclusion, but the essential factor-evaluation claim is never proved. |
| B1 | 0 | 0 | Wrong probability and no valid uncovered-area calculation. |
| B2 | 0 | 0 | Wrong bound and an unfinished repeated search. |
| B3 | 10 | 10 | Correct divergence-theorem and continuity argument. |
| B4 | 5 | 4 | Relevant Wilson/Cassini route, but the decisive residue identification is assumed. |
| B5 | 2 | 8 | Reaches the official DAG/unitriangular route; the minimal-cycle step needs a clean proof. |
| B6 | 2 | 1 | Relevant recursion, but the claimed median inequality for every distribution is false. |
| **Total** | **61** | **62** |  |

The largest calibration correction is B5, where the automated auditor missed substantial valid progress. A3 is reduced because a correct final classification still depends on a false equation as written. B6 is capped because its central general lemma has counterexamples.

The machine-readable scores are in [`manual_scores.json`](manual_scores.json). Rebuild the graph with:

```bash
python3 scripts/render_benchmark_chart.py \
  examples/qwen35_2021_dev_harness_balanced/manual_scores.json \
  assets/benchmarks/qwen35-2021-dev-harness.svg
```
