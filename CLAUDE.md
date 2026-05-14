# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clinical Knowledge Pipeline (临床指南知识抽取流水线) - A modular clinical guideline parsing toolkit for extracting v2 schema-first CQL-like typed graph knowledge from clinical guideline text using LLM-powered extraction with LSH clustering and OMOP CDM-based term standardization.

The active pipeline schema is v2 and is intentionally not compatible with the old JSON schema. Terms carry standardized clinical entities and FHIR/OMOP binding hints; predicates carry typed execution slots; rules carry `condition_dag` expression graphs. Legacy `formal_definition` and `condition` fields are deprecated trace-only fields and must not be used as executable semantics. `ConditionDAG` and `ClinicalRule` validation enforces root existence, Bool roots, unique node ids, required combine/compare operands, typed aggregates, range membership, temporal relation, and registered library function ids; temporal scope should use standard modes such as `all_time` plus reductions such as `most_recent`, not ad hoc modes like `most_recent_all_time`. Output assembly belongs in `ClinicalRule.output_assembly` or `action.output`, not as the boolean DAG root.

Langfuse tracing follows the official LangGraph/LangChain callback integration:
`graph.invoke(..., config={"callbacks": [CallbackHandler()]})`. The pipeline
builds that config via `pipeline/langfuse_tracing.py` and uses
`ChatOpenAI.with_structured_output(...)` for core LLM calls so graph nodes and
LLM generations appear under the same Langfuse trace.

## Secrets and configuration

Do not commit API keys or tokens. Keep them in process environment variables (for example `LLM_API_KEY`, `DASHSCOPE_API_KEY`) or in a local `.env` file that is listed in `.gitignore`. If `.env` was ever committed, remove it from git history (`git rm --cached`), rotate the key, and ensure teammates use their own local env files. See `.env.example` for variable names; optional `LLM_TEMPERATURE` is read by `LLMConfig.from_env()`. On first use, `get_config()` loads `PipelineConfig.from_env()` so `LLM_*` environment variables apply without a separate `set_config` call unless you override programmatically.

## Key Commands

### Running the Pipeline

**Phase 1: Extraction (Semantic IDs)**
```python
from pipeline import create_pipeline, save_to_gen

# Create pipeline (default uses Alibaba DashScope/deepseek-v4-pro)
pipeline = create_pipeline(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-v4-pro",
    preload_omop=True  # Preload OMOP concepts for post-processing
)

# Process texts
result = pipeline.run("single text")
result = pipeline.run(["text1", "text2", "text3"])  # batch processing

# Save results (all terms use semantic IDs)
save_to_gen(result)
```

**Phase 2: Post-Processing Standardization**
```python
from pipeline import run_standardization

# Run OMOP standardization with LLM review
standardization_result = run_standardization(gen_dir="gen", enable_review=True)

print(f"原始术语数: {standardization_result['original_term_count']}")
print(f"标准化后术语数: {standardization_result['final_term_count']}")
print(f"同义术语组: {standardization_result['synonym_groups']}")
```

### Stage-by-Stage Processing
```python
from pipeline import (
    extract_provenances_stage,
    cluster_provenances_stage,
    extract_terms_stage,
    extract_predicates_stage,
    extract_rules_stage,
    save_results_from_cache,
)

# Stage 1: Extract recommendations
provenances = extract_provenances_stage(texts, save_to_file=True)
# Results saved to gen/provenances.json

# Stage 2: LSH clustering
clusters, bucket_index = cluster_provenances_stage(provenances, save_to_file=True)
# Results saved to gen/clusters.json

# Stage 3: Extract terms (saves gen/terms.json, gen/med_terms.json)
extract_terms_stage(load_from_file=True, cluster_cache_path="gen/cluster_cache.json", save_results=True)

# Stage 4: Extract predicates (saves gen/predicates.json)
extract_predicates_stage(load_from_file=True, cluster_cache_path="gen/cluster_cache.json", save_results=True)

# Stage 5: Extract rules (saves gen/rules.json)
extract_rules_stage(load_from_file=True, cluster_cache_path="gen/cluster_cache.json", save_results=True)

# Aggregate and save final results
save_results_from_cache(cluster_cache_path="gen/cluster_cache.json")
```

## Architecture

### Two-Phase Design

The pipeline follows a strict two-phase architecture:

**Pipeline A: Recommendation clustering**
- Stage 1 extracts source recommendations from raw text
- Stage 2 performs LSH clustering
- Use `run_recommendation_clustering_pipeline(...)`
- Outputs `provenances.json`, `clusters.json`, legacy alias `cluster.json`, and an initial `cluster_cache.json`

**Pipeline B: Typed graph extraction**
- Stage 3 extracts terms
- Stage 4 extracts predicates
- Stage 5 extracts rules
- Use `run_graph_extraction_pipeline(...)`
- Consumes `clusters.json`/`cluster.json` and reuses `cluster_cache.json`
- This is the main debug loop after Pipeline A is fixed

**Post-Processing Binding + OMOP Standardization**
- Execute OMOP concept matching with optional LLM review
- Resolve FHIR resource/path binding from `pipeline/binding_resolver.py`
- Generate OMOP-derived candidate terminology codes; do not mark them as verified ValueSets without a terminology registry/manual review
- Generate term-OMOP mapping table
- Merge synonym terms (same OMOP concept_id)
- Update predicate and rule references

```
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 1: Extraction                          │
│                                                                 │
│  input_texts → [Provenance] → LSH_Cluster → {terms, predicates,│
│                                                rules}            │
│  (all use semantic IDs like meas.egfr, cond.ckd)               │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Phase 2: Standardization                       │
│                                                                 │
│  OMOP Matching → Binding Resolver → Synonym Merging → Updates   │
│  (FHIR paths verified by registry; OMOP codes are candidates)   │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Steps in Detail (Phase 1)

#### Step 1 — Provenance Extraction (`extract_provenances_stage`)

- **输入**: 原始文本列表 `List[str]`
- **实现**: LangGraph `StateGraph`，通过 `distribute_texts` 将每段文本以 Send API 分发给 `extract_recommendation` 节点并发执行
- **LLM 任务**: 从每段文本识别并抽取推荐意见，输出 `Provenance` 对象（含 `source`, `quote`, `recommendation_grade`, `evidence_level`, `type`, `bucket_id`）
- **参数**: `max_concurrency=5`（默认）
- **输出**: `List[Provenance]`，保存至 `gen/provenances.json`

#### Step 2 — LSH 聚类 (`cluster_provenances_stage`)

- **输入**: `List[Provenance]`
- **实现**: `lsh_cluster()` — MinHash + Locality-Sensitive Hashing
- **默认参数**: `num_bands=20`, `rows_per_band=5`, `similarity_threshold=0.5`, `max_cluster_size=8`
- **输出**: `(List[ProvenanceCluster], Dict[int, List[int]])` — 聚类列表 + bucket_index，保存至 `gen/clusters.json`

#### Step 3a — 术语抽取 (`extract_terms_stage`)

- **输入**: 聚类列表 + `cluster_cache`
- **实现**: `build_terms_extraction_graph()` — Send API fan-out，每个 cluster 由 `subgraph_extract_all_terms` / `asubgraph_extract_all_terms` 处理
- **LLM 任务**: 单次调用同时抽取 `Term`（非药物术语，含 `label`/`type`/`unit`）和 `MedicationTerm`（药物术语，含 `drug_class`/`subclass`），所有术语使用语义 ID（`meas.*`, `cond.*`, `med.*` 等）
- **输出**: `gen/terms.json`, `gen/med_terms.json`；更新 `gen/cluster_cache.json`

#### Step 3b — 谓词抽取 (`extract_predicates_stage`)

- **输入**: 聚类列表 + `cluster_cache`
- **实现**: `build_predicates_extraction_graph()` — 两步子图
  1. `extract_predicate_atoms`: 并发抽取完整 v2 typed predicate schema
  2. `assemble_predicate`: 校验并组装 `Predicates`，包含 `input_shape`, `reduction`, `final_output_type`, `temporal_scope`, `data_binding`, `library_function`, binding、unit/null/evidence 等字段
- **分发**: `distribute_predicates` 将各 cluster 的原子谓词再次 Send API 分发
- **输出**: `gen/predicates.json`；更新 `gen/cluster_cache.json`

#### Step 3c — 规则抽取 (`extract_rules_stage`)

- **输入**: 聚类列表 + `cluster_cache`（需含已抽取的谓词）
- **实现**: `build_rules_extraction_graph()` — 两步子图
  1. `extract_rule_fragments`: 并发抽取 `SimplifiedRuleItem`（含 `input_predicates`, `condition_dag`, `boolean_root`, `action`, `source_ids`）
  2. `assemble_rule_fragment`: 将 fragment 关联 provenance，组装为完整 `ClinicalRule`
- **分发**: `distribute_rule_fragments` 将 fragment 再次 Send API 分发
- **输出**: `gen/rules.json`；更新 `gen/cluster_cache.json`

> **并行替代方案**: `process_clusters_async` 一次性异步执行 3a+3b+3c（`ASYNC_CLUSTER_SUBGRAPH.ainvoke`），每个 cluster 包含术语+谓词+规则全部抽取。

#### Step 4 — 汇总保存 (`save_results_from_cache`)

- 从 `gen/cluster_cache.json` 读取各 cluster 结果，用 `merge_by_id` 去重汇总
- 分别保存 `terms.json`, `med_terms.json`, `predicates.json`, `rules.json`
- 调用 `_create_cluster_final`：优先读取 `gen/clusters.json`，兼容旧 `gen/cluster.json`，再与 `cluster_cache` 按 `cluster_id` 合并生成 `gen/cluster_final.json`

#### Step 5 — OMOP 标准化 (`run_standardization`)

内部调用 `standardize_terms()`，依次执行：

| 子步骤 | 实现 | 说明 |
|--------|------|------|
| OMOP 概念匹配 | `CombinedTermProcessor` | SPECIAL_MAPPINGS → 缩写扩展 → 中英词典 → 精确/模糊匹配 |
| LLM 审核 | `LLMTermExtractor.review_match()` | 匹配分 < `high_confidence_skip_review`（默认 92）时触发；≥ 则跳过审核 |
| 构建映射表 | `TermMappingRegistry` | 生成 `term_omop_mapping.json`（semantic_id → concept_id） |
| 同义词合并 | `build_id_replacement_map()` | 同 concept_id 的术语选 match_score 最高者为主 ID |
| 引用更新 | `update_predicate_references()` / `update_rule_references()` | 批量替换旧 ID 引用 |
| 输出 | `save_json()` | 生成 `*_standardized.json` 系列文件 |

调试第二段 graph pipeline（terms → predicates → rules）时可跳过本地 OMOP 匹配：

```python
run_standardization(gen_dir="gen/xxx", enable_review=False, skip_omop_matching=True)
```

该模式仍会执行 deterministic FHIR binding resolver、引用更新和 `*_standardized.json`
输出；OMOP code candidates 只来自已有 `term_omop_mapping.json`，不会新跑本地
fuzzy matching。

### `ClinicalGuidelinePipeline.run()` 内部数据流

`run()` 通过 `build_pipeline_graph()` 将 Step 1–3 整合为单次 `graph.invoke()` 调用：

```
input_texts
    ↓  [distribute_texts → Send API]
    ↓  extract_recommendation (并发，每文本一节点)
    ↓  do_lsh_clustering
    ↓  [route_to_clusters → Send API fan-out]
    ↓  每个 ProvenanceCluster 并发执行 CLUSTER_SUBGRAPH:
         ├─ subgraph_extract_all_terms   → terms + med_terms
         ├─ extract_predicates_subgraph  → predicates (两步法)
         └─ extract_rules_subgraph       → rules (两步法)
    ↓  [merge_by_id fan-in]
    → {terms, med_terms, predicates, rules, provenance_buffer}
```

Step 4（`save_results_from_cache`）和 Step 5（`run_standardization`）需单独调用。

### Core Data Flow
```
input_texts → [Provenance] → LSH_Cluster → [ProvenanceCluster] → {terms, med_terms, predicates, rules}
```

### Key Modules

All modules live under the `pipeline/` package:

| Module | Purpose |
|--------|---------|
| `pipeline/models.py` | Pydantic models: Term, MedicationTerm, Predicates, ClinicalRule, Provenance, AgentState, ClusterState; enums: TermLabel, Permission; reducers: merge_by_id, merge_cluster_cache_updates |
| `pipeline/config.py` | Configuration dataclasses: LLMConfig, PathConfig, MatchConfig, PipelineConfig; get_config/set_config |
| `pipeline/graph.py` | Compatibility facade that re-exports the historical `pipeline.graph` public symbols from the split graph modules |
| `pipeline/llm_factory.py` | `ChatOpenAI` construction, default LLM singleton, provider-specific kwargs such as DashScope/Qwen `enable_thinking=False` |
| `pipeline/structured_llm.py` | Shared structured-output helpers: `with_structured_output`, sync/async invoke wrappers, raw output/metadata extraction, term-stage repair retry |
| `pipeline/graph_prompts.py` | Recommendation, terms/medications, predicates, and rules extraction prompts |
| `pipeline/graph_nodes.py` | LangGraph node functions for recommendation extraction, LSH clustering, term extraction, predicate extraction, and rule extraction |
| `pipeline/graph_builders.py` | LangGraph builders, compiled cluster subgraphs, Send API fan-out routes, and cache-aware cluster processing nodes |
| `pipeline/graph_api.py` | High-level API: ClinicalGuidelinePipeline class, stage functions, process_clusters_async |
| `pipeline/processors.py` | OMOP-based processors: TermProcessor, MedicationProcessor, CombinedTermProcessor, PredicateProcessor |
| `pipeline/term_mapping.py` | TermMappingRegistry singleton: stores term-to-OMOP mappings, synonym group management |
| `pipeline/standardize_terms.py` | Post-processing: OMOP matching, synonym merging, reference updates; standardize_terms/run_standardization |
| `pipeline/lsh_cluster.py` | MinHash-based LSH clustering; lsh_cluster(provenances, num_bands, rows_per_band, similarity_threshold, max_clusters, max_cluster_size) |
| `pipeline/io_utils.py` | JSON I/O, cluster cache management, FailedTaskLogger, save_to_gen, save_stage_results |
| `pipeline/standard_library.py` | StandardLibrary singleton: loads standard/terms.json, meds.json, predicates.json and exposes v2 library functions |
| `pipeline/rate_limiter.py` | RateLimiter: max_concurrency/max_rpm/max_rps; default_rate_limiter instance (not exported in __all__) |

### State Management

- **AgentState**: Top-level state with reducers (`operator.add`, `merge_by_id`, `merge_cluster_cache_updates`)
- **ClusterState**: Per-cluster state for parallel processing
- **cluster_cache**: Persistent cache by cluster_id for resumable processing

## Data Directories

- `gen/`: Output directory for extracted knowledge and intermediate files
  - `terms.json`: Non-drug terms (semantic IDs, Phase 1 output)
  - `med_terms.json`: Medication terms (semantic IDs, Phase 1 output)
  - `predicates.json`: Logical predicates (Phase 1 output)
  - `rules.json`: Clinical rules (Phase 1 output)
  - `provenances.json`: Raw recommendations
  - `clusters.json`: LSH clustering results
  - `cluster_cache.json`: Per-cluster extraction cache
  - `cluster_final.json`: Merged final clusters
  - `failed_tasks.json`: Failed task log for retry
  - `term_omop_mapping.json`: Term-to-OMOP concept mapping (Phase 2)
  - `terms_standardized.json`: Standardized terms with OMOP concept IDs (Phase 2)
  - `med_terms_standardized.json`: Standardized medication terms (Phase 2)
  - `predicates_standardized.json`: Predicates with updated references (Phase 2)
  - `rules_standardized.json`: Rules with updated references (Phase 2)
- `standard/`: Standard library directory (loaded by StandardLibrary singleton)
  - `terms.json`, `meds.json`, `predicates.json`
- `omop_normalizer/`: OMOP CDM-based term standardization module
  - `matcher.py`: OMOPMatcher for concept matching (SPECIAL_MAPPINGS, ABBREVIATION_DICT, CHINESE_ENGLISH_DICT strategies)
  - `cache.py`: MappingCache for SQLite-based caching (tables: term_mapping, review_queue)
  - `normalizer.py`: ChineseTermNormalizer; create_normalizer(concept_csv_path=...)
  - `extractor.py`: LLMTermExtractor for match review (EXTRACTION_PROMPT, REVIEW_PROMPT)
  - `models.py`: Data models (ExtractedTerm, ConceptMatch, MappingResult, ReviewResult, TermMapping)
  - `dictionaries.py`: ABBREVIATION_DICT, CHINESE_ENGLISH_DICT, DRUG_CLASS_MEMBERS, SPECIAL_MAPPINGS, DOMAIN_MAPPING
  - `cli.py`: Command-line interface
- `term_mapping_cache.db`: SQLite cache for term mappings

## LLM Configuration

Default configuration uses Alibaba DashScope (deepseek-v4-pro). Configure via:

```python
from pipeline import set_config, PipelineConfig, LLMConfig, MatchConfig

# LLM configuration
config = PipelineConfig(llm=LLMConfig(
    api_key="your-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-v4-pro",
    temperature=0.1,
    max_tokens=16384,
    timeout=300.0,   # default: 300.0 (5 min, DashScope can be slow)
    max_retries=3
))

# Match configuration (OMOP + LLM review)
config = PipelineConfig(match=MatchConfig(
    term_threshold=85.0,                # Minimum OMOP match score for terms（默认略提高以减少低分误配）
    med_threshold=85.0,                 # Minimum drug match score（Drug 域在 CombinedTermProcessor 中单独使用）
    predicate_fuzzy_threshold=70.0,     # Fuzzy match threshold for predicates
    predicate_high_confidence=90.0,     # High-confidence threshold for predicates
    enable_review=True,                 # Enable LLM review for matches
    review_threshold=95.0,              # Trigger review below this score
    high_confidence_skip_review=92.0    # 匹配分≥此值则跳过 LLM 审核（默认 92，减少审核调用）
))

set_config(config)
```

Or via environment variables: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`

### Alternative Providers

```python
# OpenRouter
pipeline = create_pipeline(
    api_key="your-key",
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-v3.1-terminus"
)

# DeepSeek direct
pipeline = create_pipeline(
    api_key="your-key",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)
```

## Dependencies

- `langchain-openai`: LLM interface (OpenAI-compatible)
- `langgraph`: Workflow orchestration with Send API for parallel execution
- `pydantic`: Data validation and serialization
- `rapidfuzz`: Fuzzy string matching (used in OMOP matcher)
- `pandas`: Data manipulation for OMOP concept loading
- `openpyxl`: Excel file reading for ValueSets
- `openai`: Transitive/compatibility dependency used by `langchain-openai`
- `langfuse`: LangGraph/LangChain tracing via `CallbackHandler`

## Structured Output Implementation

The pipeline uses LangChain `ChatOpenAI.with_structured_output(...)` for core structured LLM calls. Graph invocation config includes Langfuse's LangChain `CallbackHandler`, so LangGraph nodes and LLM generations are captured under the same trace.

Implementation notes for the split graph modules:

- `pipeline/langfuse_tracing.py`: loads local `.env`, normalizes `LANGFUSE_BASE_URL` to `LANGFUSE_HOST`, creates `CallbackHandler`, builds LangGraph config, and flushes traces.
- `pipeline/llm_factory.py`: `_get_default_llm()` / `_create_llm()` create `ChatOpenAI` clients; `_chat_openai_kwargs()` adds DashScope/Qwen-specific `extra_body={"enable_thinking": False}` so tool-calling structured output works with qwen3-style models.
- `pipeline/structured_llm.py`: `_invoke_structured_output_direct()` / `_ainvoke_structured_output_direct()` are historical names retained for compatibility, but internally they now call LangChain structured output; `_invoke_structured_list()` / `_ainvoke_structured_list()` wrap list-typed extraction outputs, parsing errors, and term-stage repair retry.
- `pipeline/graph_prompts.py`: owns all extraction prompts.
- `pipeline/graph_nodes.py`: owns node implementations and calls `structured_llm`.
- `pipeline/graph_builders.py`: owns compiled LangGraph graphs/subgraphs and cache-aware cluster processing.
- `pipeline/graph.py`: re-export facade for older imports; do not add new implementation there unless preserving compatibility.

## Async I/O Support

The pipeline supports **true async I/O** (not `asyncio.to_thread` wrapping) for LLM API calls.

### Async Node Functions

All three extraction node functions have async versions:

| Sync Function | Async Function | Purpose |
|---------------|----------------|---------|
| `subgraph_extract_all_terms` | `asubgraph_extract_all_terms` | Terms + medications extraction |
| `extract_predicates_subgraph` | `async_extract_predicates_subgraph` | Predicates extraction |
| `extract_rules_subgraph` | `async_extract_rules_subgraph` | Rules extraction |
| `process_cluster` | `aprocess_cluster` | Full cluster processing |

Note: `subgraph_extract_all_terms` (sync) and `extract_predicates_subgraph` / `extract_rules_subgraph` (sync) live in `graph_nodes.py`; `pipeline.graph` re-exports them for historical imports, but they are intentionally not exported from `pipeline/__init__.py`'s `__all__`.

### Async Subgraphs

```python
from pipeline import CLUSTER_SUBGRAPH, ASYNC_CLUSTER_SUBGRAPH

# Sync version
result = CLUSTER_SUBGRAPH.invoke(state)

# Async version (true async I/O)
result = await ASYNC_CLUSTER_SUBGRAPH.ainvoke(state)
```

### Async Stage Functions

```python
from pipeline import process_clusters_async

# LangChain + LangGraph ainvoke（高并发 LLM 调用）
result = await process_clusters_async(clusters=clusters, max_concurrency=10)
```

### Implementation Details

- Uses `ChatOpenAI.with_structured_output(...).ainvoke(...)` through LangChain.
- LangGraph config, including Langfuse callbacks, is propagated through async subgraphs and LLM calls.
- `process_clusters_async` 使用 `ASYNC_CLUSTER_SUBGRAPH.ainvoke`，为真异步 I/O。

### When to Use Async Cluster Processing

- High-concurrency LLM API calls
- Integration with other async code (FastAPI, asyncio event loops)
- Non-blocking I/O without thread pool overhead

## Term Standardization (Phase 2)

OMOP CDM-based term standardization runs as a **post-processing step** after the extraction pipeline completes.

### Quality tuning（抽取质量 + OMOP 准度 + 审核次数）

| 目标 | 实现要点 |
|------|----------|
| **更高质量术语** | `graph_prompts.py` 中 `TERMS_AND_MEDS_PROMPT`：要求原子概念、英文可映射名、禁止模糊/整句、修正药物 id 示例为 `med.xxx`；`LLMConfig.temperature` 默认 **0.05** 提高稳定性。 |
| **更准/更快的 OMOP 匹配** | `processors._preprocess_name_for_omop`：空白规范化 + 中英词典长词优先替换；`CombinedTermProcessor` 在 **Drug** 域使用 `med_threshold`，其它域使用 `term_threshold`；`OMOPMatcher` 按 domain 缩小 contains/fuzzy/multi-keyword 候选集。 |
| **提高阈值、减少误配** | `MatchConfig.term_threshold` / `med_threshold` 默认 **85**（原 80），低分候选不注册映射。 |
| **减少 LLM 审核** | `high_confidence_skip_review` 默认 **92**（原 98）：匹配分 ≥ 92 即跳过审核；若仍希望「宁可多审」可把该值调回 96–98。 |
| **专科词表** | 在 `omop_normalizer/dictionaries.py` 的 `CHINESE_ENGLISH_DICT` / `SPECIAL_MAPPINGS` 中补充机构常用译名，可显著提升命中率与分数。 |
| **可信 binding 边界** | `binding_resolver.py` 将 LLM binding 降为 `candidate_binding`；FHIR resource/path 来自本地 registry；OMOP code 只作为 candidate，等待 terminology registry/manual review。 |

### Key Principle
Semantic IDs are preserved throughout extraction. OMOP matching happens only during standardization:
- Extraction: `Term(id="meas.egfr", name="eGFR")` → saved to `terms.json`
- Standardization: OMOP matching generates `term_omop_mapping.json`
- Final output: `terms_standardized.json` with `candidate_binding`, `verified_binding`, and `binding_evidence`

### Components

1. **OMOPMatcher** (`omop_normalizer/matcher.py`): Matches extracted terms to OMOP concepts
   - Domain filtering via `RELEVANT_DOMAINS`, `DRUG_CLASS_KEYWORDS`
   - `match(english_term, domain_id=None, top_k=...)` 
   - Matching strategies: SPECIAL_MAPPINGS → abbreviation expansion → Chinese-English dict → exact/fuzzy

2. **MappingCache** (`omop_normalizer/cache.py`): SQLite-based cache (`term_mapping_cache.db`)
   - Tables: `term_mapping`, `review_queue`

3. **Processors** (`pipeline/processors.py`):
   - `TermProcessor`: Non-drug terms (Measurement, Condition, Procedure, Observation)
   - `MedicationProcessor`: Drug terms (Drug domain)
   - `CombinedTermProcessor`: Handles both with intelligent cross-domain search; `_match_in_omop`, `_review_match`
   - `PredicateProcessor`: Predicate deduplication/standardization
   - Convenience functions: `process_terms`, `process_med_terms`, `process_combined_terms`, `process_predicates`

4. **LLM Review** (`omop_normalizer/extractor.py`): Automated validation of OMOP matches
   - Triggered when match_score < `high_confidence_skip_review` (default: 92)
   - Uses LLM to verify semantic correctness between original term and matched concept
   - Rejects incorrect matches, preserving original terms
   - Configurable via `MatchConfig`:
     - `enable_review`: Enable/disable LLM review (default: True)
     - `review_threshold`: Minimum score to trigger review (default: 95)
     - `high_confidence_skip_review`: Skip review at or above this score (default: 92)

5. **Thread Safety**: OMOP components use thread-safe singleton pattern
   - `threading.Lock` with double-checked locking
   - Preload via `create_pipeline(preload_omop=True)` recommended for parallel execution
   - Prevents duplicate loading in multi-threaded environments

6. **TermMappingRegistry** (`pipeline/term_mapping.py`): Singleton registry
   - `TermOMOPMapping` dataclass: `semantic_id`, `concept_id`, `concept_name`, `domain_id`, `vocabulary_id`, `match_score`, `match_type`
   - Methods: `register_term/med`, `get_term/med_mapping`, `get_term/med_synonyms`, `get_all_term/med_groups`, `get_primary_term/med_id`, `save`, `load`, `clear`
   - Properties: `term_count`, `med_count`, `synonym_group_count`

7. **Binding Resolver** (`pipeline/binding_resolver.py`):
   - Maintains deterministic FHIR resource/path registry for measurement, condition, procedure, observation, and medication terms
   - Writes registry-derived FHIR paths to `verified_binding.fhir_binding`
   - Moves LLM binding suggestions and OMOP-derived codes into `candidate_binding`
   - Adds `binding_evidence` so executor/evaluator can distinguish LLM hints, local OMOP matches, and registry-derived bindings

## Permission Types

The `Permission` enum (in `pipeline/models.py`) defines rule action types:
- Usage: `allow`, `recommend`, `require`, `caution`, `avoid`, `contraindicate`, `continue`, `stop`, `consider`
- Dose adjustment: `reduce_dose`, `increase_dose`, `start_low_dose`, `max_dose_limit`, `titrate`, `maintain_dose`
- Class methods: `usage_permissions()`, `dose_permissions()`, `priority_order()`, `dose_priority_order()`
- Instance methods: `priority()`, `is_restrictive()`, `is_permissive()`, `is_dose_adjustment()`

**Note**: `RuleAction.permission` (used for LLM structured output) is a `Literal` type that includes all values **except** `maintain_dose`. The `Permission` enum is authoritative for domain logic; `RuleAction` is used for structured extraction.

## Key Classes

### ClinicalGuidelinePipeline
Main entry point for the pipeline:
```python
from pipeline import ClinicalGuidelinePipeline, create_pipeline

pipeline = ClinicalGuidelinePipeline()
result = pipeline.run(texts, max_concurrency=5)

# Or use the factory function
pipeline = create_pipeline(api_key="...", model="deepseek-v4-pro", preload_omop=True)
```

There is **no** `run_async` method on `ClinicalGuidelinePipeline`; use `process_clusters_async` for async cluster processing.

### ProvenanceCluster
Represents a cluster of similar recommendations (Pydantic BaseModel, not dataclass):
```python
class ProvenanceCluster(BaseModel):
    cluster_id: int
    provenances: List[Provenance]
    texts_formatted: List[str] = []
```

### AgentState
Top-level state with automatic deduplication:
```python
class AgentState(TypedDict):
    input_texts: List[str]
    messages: Annotated[Sequence[AnyMessage], add_messages]
    provenance_buffer: Annotated[List[Provenance], operator.add]
    clusters: List[ProvenanceCluster]
    terms: Annotated[List[Term], merge_by_id]           # auto-dedup by id
    med_terms: Annotated[List[MedicationTerm], merge_by_id]
    predicates: Annotated[List[Predicates], merge_by_id]
    rules: Annotated[List[ClinicalRule], merge_by_id]
    cluster_cache_updates: Annotated[Dict[int, Dict[str, Any]], merge_cluster_cache_updates]
    cluster_cache: Dict[int, Dict[str, Any]]
```

### Core Data Models

```python
class Term(BaseModel):
    id: str
    name: str
    label: TermLabel
    type: str                 # Bool / Quantity / Code / DateTime / Interval / Enum
    clinical_entity: str
    concept: str
    value_set_binding: Optional[CodeBinding]
    code_bindings: List[CodeBinding]
    data_bindings: DataBinding
    fhir_binding_hint: Dict[str, Any]
    omop_binding_hint: Dict[str, Any]
    binding_status: Literal["candidate", "verified", "rejected", "unresolved"]
    candidate_binding: Dict[str, Any]
    verified_binding: Optional[BindingResolution]
    binding_evidence: List[BindingEvidence]
    source_evidence: List[SourceEvidence]
    normalization_confidence: Optional[float]

class MedicationTerm(BaseModel):
    id: str
    name: str
    drug_class: Optional[str] = Field(None, alias="class")
    subclass: Optional[str] = None

class Predicates(BaseModel):
    id: str
    name: str
    entity: str
    entity_type: str
    aspect: str
    input_shape: str
    reduction: ReductionSpec
    return_type: str
    final_output_type: str
    temporal_scope: TemporalScope
    data_binding: DataBinding
    library_function: List[str]
    null_policy: str
    evidence: List[SourceEvidence]

class ClinicalRule(BaseModel):
    id: str
    label: str
    input_predicates: List[str]
    condition_dag: ConditionDAG
    boolean_root: str
    action: Action
    missing_data_policy: str
    provenance: List[Provenance] = []

class Action(BaseModel):
    subjects: List[str]
    permission: Permission
    strength: Optional[str]
    intent: Optional[str]
    timing: Optional[Dict[str, Any]]
    monitoring: List[str]
    requirements: List[str] = []
```

Minimum v2 library functions:

- `lib.fhir.filter_by_status`
- `lib.fhir.filter_by_lookback`
- `lib.fhir.most_recent`
- `lib.fhir.count_in_window`
- `lib.fhir.count_all_time`
- `lib.fhir.extremum`
- `lib.fhir.concept_in_valueset`

## Error Handling

- **FailedTaskLogger**: Singleton that logs failed tasks to `gen/failed_tasks.json`
- Supports retry of failed extractions
- Automatic logging of cluster processing failures
- Access via `get_failed_task_logger()` or `FailedTaskLogger` (both exported)

## Test Scripts

- Default pytest collection is limited to `tests/integration`, so it targets the real pipeline/API-facing test area.
- Non-real-LLM tests are isolated under `tests/isolated_non_llm/` and are opt-in:
  `python -m pytest tests/isolated_non_llm`
- `test_sglt2i.py`: Full two-phase test (extraction + standardization) using SGLT2i_zh.xlsx
- `scripts/manual_tests/run_pipeline_stages.py`: Manual stage-by-stage test runner
- `scripts/manual_tests/run_term_extraction.py`: Term extraction manual test
