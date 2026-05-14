# v2 Pipeline Output

Generated from `SGLT2i_zh.xlsx`, using the first 5 recommendations from each of the guideline and consensus sheets.

## Counts

- Provenances: 10
- Clusters: 3
- Terms: 22
- Medication terms: 13
- Predicates: 32
- Rules: 20

## Main Files

- `summary.json`: run summary and v2 schema checks
- `provenances.json`: input recommendation quotes used for this run
- `clusters.json`: LSH cluster output
- `cluster_cache.json`: per-cluster cached stage outputs
- `terms.json`: v2 term stage output
- `med_terms.json`: v2 medication term stage output
- `predicates.json`: v2 typed predicate stage output
- `rules.json`: v2 rule stage output with `condition_dag`
- `cluster_final.json`: clusters merged with extracted v2 outputs
- `failed_tasks.json`: one logged malformed LLM term output during extraction; later predicate/rule stages still produced valid v2 outputs

## v2 Checks

The generated `summary.json` reports:

- terms have `clinical_entity`
- med terms have `clinical_entity`
- predicates have `input_shape`, `reduction`, and `final_output_type`
- rules have `condition_dag`
- rule DAG roots resolve to `Bool`
