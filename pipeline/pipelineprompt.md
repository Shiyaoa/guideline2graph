# Guideline2Graph Pipeline v2 Prompts

This pipeline is now a non-compatible v2 schema-first CQL-like typed graph extraction pipeline. The source of truth is `planv2.md`; `CQL_notes.md` informs CQL semantics; `guideline2graph_work_basis.md` is background only.

## Stage Contract

The pipeline still runs three extraction stages:

1. `terms -> TermExtractionResult`
2. `predicates -> PredicateExtractionBatch`
3. `rules -> SubmitSimplifiedRules`

The core output is no longer legacy `formal_definition` or `condition` text. Those fields may exist only as deprecated trace fields. Executable semantics must be carried by typed predicate fields and `condition_dag`.

## Terms Stage

The terms prompt asks for standardized clinical entities and medication entities with traceability.

Required non-drug term fields:

- `id`, `name`, `label`, `type`
- `clinical_entity`
- `concept`
- `value_domain`
- `value_set_binding` or `code_bindings` as candidate bindings only
- `data_bindings`
- `fhir_binding_hint`
- `omop_binding_hint`
- `binding_status` as `candidate` or `unresolved`
- `source_evidence`
- `normalization_confidence`

Required medication fields:

- `id`, `name`
- `class`, `subclass`
- `clinical_entity`
- `concept`
- `value_set_binding` or `code_bindings` as candidate bindings only
- `data_bindings`
- `fhir_binding_hint`
- `omop_binding_hint`
- `binding_status` as `candidate` or `unresolved`
- `source_evidence`
- `normalization_confidence`

The LLM is not the authority for executable binding. Exact codes, ValueSets,
FHIR paths, and OMOP hints produced by the terms prompt are candidate evidence
only. The post-processing resolver adds:

- `candidate_binding`: LLM suggestions plus OMOP-derived candidate code bindings.
- `verified_binding`: deterministic FHIR resource/path mapping from the local registry.
- `binding_evidence`: audit trail for LLM, OMOP, registry, and future manual/terminology validation.

FHIR resource/path binding is maintained in `binding_resolver.py`. OMOP matching
may generate candidate terminology codes, but those are not final verified
FHIR ValueSets until a terminology registry or manual review validates them.

## Predicates Stage

The predicates prompt asks for complete typed predicate schemas. A predicate describes a reusable CQL-like observation pattern, including retrieve/filter/reduce/extract/compare slots when applicable.

Every predicate should include:

- `id`, `name`, `description`, `source_text`
- `entity`, `entity_type`, `aspect`
- `input_shape`
- `reduction`
- `return_type`
- `final_output_type`
- `temporal_scope`
- `data_binding`
- `library_function`
- `value_set_binding` or `code_binding`, inherited from available terms or explicit source text only
- `unit` and `quantity_semantics`
- `range_spec` when `aspect == "quantity_range"` or the predicate id ends with `.range`
- `null_policy`
- `evidence`, `source_span`
- `dependencies`

Canonical count rules are not represented as Bool predicates. A count scenario should expose a predicate returning `List<Resource>` and let the rule use `aggregate(count)` followed by `compare`.

Canonical temporal modes are `all_time`, `currently_active`, `any_in_lookback`, `most_recent_in_lookback`, `extremum_in_lookback`, `count_in_lookback`, `count_all_time`, and `relative_to_event`. Use `reduction.operator = "most_recent"` with `temporal_scope.mode = "all_time"` instead of inventing `most_recent_all_time`.

Range predicates are schema-only in v2: `range_spec.intervals` must be non-empty, interval ids must be unique, and bounds must be legal. Do not add Z3-specific requirements or interval-reasoning outputs.

## Rules Stage

The rules prompt asks for `ClinicalRule` fragments with typed `condition_dag`.

Every rule should include:

- `id`, `label`
- `input_predicates`
- `condition_dag`
- `boolean_root`
- `missing_data_policy`
- `action`
- `priority`
- optional `output_assembly`
- `source_ids`

Supported DAG node types include:

- `predicate_ref`
- `combine` with `and`, `or`, `not`
- `compare`
- `aggregate`, especially `count`
- `coalesce`
- `temporal_relation`
- `range_membership`
- `library_function`
- `output_assembly`

Each DAG node must declare `return_type`. The root node must resolve to `Bool` for ordinary clinical rules. Validation rejects duplicate node ids, missing roots, non-Bool roots, `combine` nodes without `inputs`, `compare` nodes without `left`/`right`, `coalesce` nodes with fewer than two inputs, malformed `range_membership` and `temporal_relation` nodes, unregistered `library_function` ids, and aggregates with incompatible return types.

Output assembly belongs in rule-level `output_assembly` or `action.output`; it should not be made the boolean `condition_dag.root`. Supported `output_assembly.operator` values are `union_except_null`, `union`, `except_null`, and `message_list`.

## Standard Library Functions

The minimum v2 library functions are implemented in `standard_library.py`:

- `lib.fhir.filter_by_status`
- `lib.fhir.filter_by_lookback`
- `lib.fhir.most_recent`
- `lib.fhir.count`
- `lib.fhir.count_in_window`
- `lib.fhir.count_all_time`
- `lib.fhir.extremum`
- `lib.fhir.concept_in_valueset`
- `lib.fhir.condition.active`
- `lib.fhir.medication_statement.active`
- `lib.fhir.medication_order.active`
- `lib.fhir.procedure.completed`
- `lib.fhir.observation.verified`

Resource-specific wrappers such as active condition or completed procedure can be represented through `library_function` references or predicate `filters` until an execution engine expands them.

## Traceability

All stages should preserve source traceability through:

- `source_text`
- `source_span`
- `source_chunk`
- `source_evidence`
- `provenance`

Trace fields are not executable semantics; they support audit and review.
