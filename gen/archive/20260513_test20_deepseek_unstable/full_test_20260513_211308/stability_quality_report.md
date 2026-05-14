# Test20 DeepSeek Pipeline Stability & Quality Report

Run directory: `D:\WAICA\guideline2graph\gen\full_test_20260513_211308`  
Input: `test.xlsx`, first 20 rows  
Model: `deepseek-v4-pro` via `.env` / printed startup config  
Mode: `--skip-omop-matching`, `--max-concurrency 3`  
Generated: 2026-05-13 21:13:08

## Runtime

| Stage | Seconds |
|---|---:|
| 读取 Excel | 0.2 |
| 初始化 Pipeline + OMOP 预加载 | 0.6 |
| Step 1 - Provenance 抽取 | 37.9 |
| Step 2 - LSH 聚类 | 0.2 |
| Step 3 - 异步知识抽取 (术语+谓词+规则) | 613.9 |
| Step 4 - 汇总保存 cluster_final.json | 0.0 |
| Step 5 - OMOP 标准化 | 0.0 |
| **Total** | **652.8** |


## Output Counts

| Artifact | Count |
|---|---:|
| provenances | 31 |
| clusters | 6 |
| terms_standardized | 24 |
| med_terms_standardized | 11 |
| predicates_standardized | 59 |
| rules_standardized | 31 |

## Stability Findings

- Process exit: success.
- Effective model: startup log shows `deepseek-v4-pro`; the earlier `qwen3-max` issue was caused by a hard-coded test entrypoint and has been fixed.
- Failed-task logging in this run was polluted by stale global `gen/failed_tasks.json`: `run_timing.json` says 43 failures, but 42 were old entries from earlier runs and 1 was from this run.
- Current run failure: `extract_all_terms` / `API 调用失败: Connection error.`.
- Langfuse export had post-run read timeouts. Graph artifacts were written, but trace completeness in Langfuse is not reliable for this run.
- Cluster completeness: 3 / 6 clusters produced predicates without term coverage.

### Cluster Cache Shape

| Cluster | Terms | Med Terms | Predicates | Rules |
|---:|---:|---:|---:|---:|
| 0 | 8 | 8 | 7 | 4 |
| 1 | 0 | 0 | 10 | 4 |
| 2 | 0 | 0 | 15 | 4 |
| 3 | 9 | 2 | 11 | 4 |
| 4 | 0 | 0 | 11 | 7 |
| 5 | 10 | 9 | 19 | 8 |


## Schema And Computability Checks

| Check | Result |
|---|---:|
| Pydantic validation errors | 0 |
| Bad DAG roots / non-Bool roots | 0 |
| Bad DAG inputs | 0 |
| Direct predicate inputs in combine nodes | 60 |
| Node-to-node inputs | 15 |
| Missing predicate entity refs | 21 |
| Missing action subject refs | 11 |
| Terms with verified FHIR binding | 24/24 |
| Med terms with verified FHIR binding | 11/11 |
| Terms with OMOP mapping candidates | 0/24 |

DAG node type distribution: `{'predicate_ref': 12, 'combine': 34}`

Predicate output types: `{'Bool': 56, 'Enum': 3}`  
Temporal modes: `{'all_time': 43, 'currently_active': 14, 'relative_to_event': 1, 'any_in_lookback': 1}`  
Action permissions: `{'recommend': 19, 'maintain_dose': 2, 'caution': 1, 'stop': 1, 'require': 2, 'avoid': 5, 'reduce_dose': 1}`

## Quality Assessment

### Strengths

- The delivered JSON is structurally valid v2 output: terms, meds, predicates, and rules all parse under the current Pydantic models.
- Rule roots are valid Bool nodes, and every DAG input resolves either to an in-DAG node or a generated predicate id.
- Standardized term outputs carry deterministic FHIR resource/path binding from the local registry.
- The rule set captures many clinically important SGLT2i recommendation types: T2DM/CKD/eGFR, HF phenotypes, perioperative stopping, pregnancy/lactation avoidance, euglycemic DKA, and renal thresholds.

### Main Quality Risks

- Term coverage is incomplete. Clusters 1, 2, and 4 have predicates/rules but no term objects, so graph execution cannot ground all predicate entities.
- 21 predicates point to entities absent from the standardized term/med vocabularies. Examples: `[('pred.cond.hfmref.exists', 'cond.hfmref'), ('pred.cond.hfpef.exists', 'cond.hfpef'), ('pred.meas.serum_creatinine.increase_pct.lt.50', 'meas.serum_creatinine'), ('pred.meas.serum_creatinine.value.le.221', 'meas.serum_creatinine'), ('pred.meas.serum_creatinine.increase_pct.range.50_100', 'meas.serum_creatinine'), ('pred.meas.serum_creatinine.value.range.221_309', 'meas.serum_creatinine'), ('pred.meas.serum_creatinine.increase.gt.100pct', 'meas.serum_creatinine'), ('pred.meas.serum_creatinine.value.gt.309', 'meas.serum_creatinine')]`.
- 11 rule action subjects are absent from med terms. Examples: `[('rule.t2dm_ascvd_or_high_risk_glp1ra_or_sglt2i', 'med.class.glp1ra_ascvd_benefit'), ('rule.t2dm_ckd_sglt2i_or_glp1ra_ckd_benefit', 'med.class.sglt2i_ckd_benefit'), ('rule.t2dm_ckd_sglt2i_or_glp1ra_ckd_benefit', 'med.class.glp1ra_ckd_benefit'), ('rule.symptomatic_hfref_sglt2i_recommend', 'med.dapagliflozin'), ('rule.symptomatic_hfref_sglt2i_recommend', 'med.empagliflozin'), ('rule.symptomatic_hfref_sglt2i_require', 'med.dapagliflozin'), ('rule.symptomatic_hfref_sglt2i_require', 'med.empagliflozin'), ('rule.egfr_lt_20_avoid_empagliflozin', 'med.empagliflozin')]`. This is clinically important for dapagliflozin/empagliflozin-specific rules.
- Rule traceability is weak: rules currently do not carry direct `source_text` / `source_span`; provenance exists upstream but is not copied onto rule objects.
- DAG expressiveness is shallow in this run: no aggregate/compare/coalesce/temporal_relation nodes appeared. Numeric thresholds are mostly precompiled into Bool predicates, which is computable but less inspectable than the intended CQL-like expression graph.
- OMOP matching was intentionally skipped, so terminology code candidates are not present. FHIR structural binding is available, but terminology binding is not delivery-grade.

## Verdict

- Stability: **partial pass** for a 20-row smoke run. The pipeline completed and wrote artifacts, but one API connection failure and incomplete term coverage show it is not robust enough for unattended batch delivery.
- Schema correctness: **pass**.
- Computability: **conditional pass**. Rule DAGs are Bool-resolvable, but missing entity/action term references reduce executable coverage.
- Clinical delivery quality: **not yet production-ready**. Suitable for debugging and human review; not suitable for downstream executor or clinical semantic evaluation without repairing term coverage, action subject grounding, and rule traceability.

## Recommended Next Fixes

1. Add per-cluster retry for term extraction before predicate/rule extraction, especially when terms are empty but provenances are non-empty.
2. Enforce referential integrity after each cluster: every predicate `entity` and every action subject must exist in terms/med_terms or be auto-created as a minimal term with source evidence.
3. Copy provenance quote/source_span into each assembled `ClinicalRule`.
4. Add a graph QA gate that fails the run when missing entity/action refs exceed zero.
5. Run full OMOP matching only as a separate terminology-finalization job after graph QA passes.
