"""v2 schema-first typed graph validation tests."""
import json

import pytest
from pydantic import ValidationError

from pipeline.models import (
    TermExtractionResult,
    TemporalScope,
    PredicateExtractionBatch,
    SubmitSimplifiedRules,
    ClinicalRule,
    ConditionDAG,
)
from pipeline.standard_library import get_v2_library_functions


def _root_node(rule: ClinicalRule):
    by_id = {node.id: node for node in rule.condition_dag.nodes}
    return by_id[rule.condition_dag.root]


def _assert_dag_parseable(rule: ClinicalRule):
    assert rule.condition_dag.root
    by_id = {node.id: node for node in rule.condition_dag.nodes}
    assert rule.condition_dag.root in by_id
    assert _root_node(rule).return_type == "Bool"
    for pred_id in rule.input_predicates:
        assert pred_id.startswith("pred.")


def test_terms_stage_v2_json_schema():
    payload = {
        "terms": [
            {
                "id": "meas.egfr",
                "name": "Estimated glomerular filtration rate",
                "label": "measures",
                "type": "Quantity",
                "clinical_entity": "meas.egfr",
                "concept": "Estimated glomerular filtration rate",
                "value_domain": "Quantity",
                "unit": "mL/min/1.73m2",
                "value_set_binding": {"type": "Code", "system": "LOINC", "code": "33914-3", "confidence": 0.8},
                "data_bindings": {
                    "FHIR": {"resource": "Observation", "value_path": "Observation.valueQuantity"},
                    "OMOP": {"table": "measurement", "value_path": "value_as_number"},
                },
                "source_evidence": [{"source_text": "eGFR >= 20"}],
                "normalization_confidence": 0.9,
            }
        ],
        "med_terms": [
            {
                "id": "med.class.sglt2i",
                "name": "SGLT2 inhibitors",
                "class": None,
                "subclass": None,
                "clinical_entity": "med.class.sglt2i",
                "concept": "SGLT2 inhibitor drug class",
                "value_set_binding": {"type": "ValueSet", "name": "SGLT2 inhibitors", "confidence": 0.7},
                "data_bindings": {"FHIR": {"resource": "MedicationStatement"}, "OMOP": {"table": "drug_exposure"}},
                "source_evidence": [{"source_text": "SGLT2i"}],
                "normalization_confidence": 0.9,
            }
        ],
    }

    result = TermExtractionResult.model_validate(payload)
    dumped = json.loads(result.model_dump_json(by_alias=True))
    assert dumped["terms"][0]["clinical_entity"] == "meas.egfr"
    assert dumped["terms"][0]["data_bindings"]["FHIR"]["resource"] == "Observation"
    assert dumped["med_terms"][0]["value_set_binding"]["type"] == "ValueSet"


def test_predicate_stage_v2_typed_schema():
    payload = {
        "predicates": [
            {
                "id": "pred.meas.egfr.value.ge.20",
                "name": "Most recent eGFR >= 20",
                "description": "Most recent eGFR is at least 20.",
                "source_text": "eGFR >= 20",
                "entity": "meas.egfr",
                "entity_type": "observation",
                "aspect": "quantity",
                "input_shape": "List<Observation>",
                "reduction": {"operator": "most_recent", "output_type": "Quantity"},
                "return_type": "Bool",
                "final_output_type": "Bool",
                "temporal_scope": {"mode": "all_time"},
                "data_binding": {"FHIR": {"resource": "Observation"}, "OMOP": {"table": "measurement"}},
                "library_function": ["lib.fhir.most_recent"],
                "value_set_binding": {"type": "Code", "system": "LOINC", "code": "33914-3"},
                "unit": "mL/min/1.73m2",
                "quantity_semantics": {"unit": "mL/min/1.73m2", "comparator": "ge", "value": 20},
                "retrieve": {"resource": "Observation", "code_binding": "meas.egfr"},
                "extract": {"path": "valueQuantity", "type": "Quantity"},
                "compare": {"operator": "ge", "value": 20, "unit": "mL/min/1.73m2"},
                "null_policy": "unknown",
                "evidence": [{"source_text": "eGFR >= 20"}],
                "source_span": {"source_text": "eGFR >= 20"},
                "dependencies": ["meas.egfr"],
            }
        ]
    }

    batch = PredicateExtractionBatch.model_validate(payload)
    pred = batch.predicates[0]
    assert pred.input_shape == "List<Observation>"
    assert pred.reduction.operator == "most_recent"
    assert pred.final_output_type == "Bool"
    assert pred.formal_definition is None


def test_standard_library_minimum_function_set():
    function_ids = {fn.id for fn in get_v2_library_functions()}
    assert {
        "lib.fhir.filter_by_status",
        "lib.fhir.filter_by_lookback",
        "lib.fhir.most_recent",
        "lib.fhir.count_in_window",
        "lib.fhir.count_all_time",
        "lib.fhir.extremum",
        "lib.fhir.concept_in_valueset",
    }.issubset(function_ids)


def test_rule_dag_three_new_rules_examples():
    payload = {
        "rules": [
            {
                "id": "rule.dm_with_ascvd_or_high_risk_sglt2i_recommend",
                "label": "T2DM with ASCVD or high ASCVD risk recommends SGLT2i",
                "input_predicates": ["pred.cond.t2dm.exists", "pred.cond.ascvd.exists", "pred.cond.ascvd.risk.eq.High"],
                "condition_dag": {
                    "nodes": [
                        {"id": "N1", "type": "combine", "operator": "or", "inputs": ["pred.cond.ascvd.exists", "pred.cond.ascvd.risk.eq.High"], "return_type": "Bool"},
                        {"id": "ROOT", "type": "combine", "operator": "and", "inputs": ["pred.cond.t2dm.exists", "N1"], "return_type": "Bool"},
                    ],
                    "root": "ROOT",
                },
                "action": {"subjects": ["med.class.sglt2i"], "permission": "recommend", "strength": "strong", "intent": "initiate_or_continue"},
                "source_ids": "q1",
            },
            {
                "id": "rule.renal_mild_impairment_continue_sglt2i",
                "label": "Renal mild impairment continues SGLT2i",
                "input_predicates": ["pred.meas.serum_creatinine.delta.range", "pred.meas.serum_creatinine.value.range", "pred.meas.egfr.value.range"],
                "condition_dag": {
                    "nodes": [
                        {"id": "N1", "type": "range_membership", "input": "pred.meas.serum_creatinine.delta.range", "value": "scr_delta.50_100", "return_type": "Bool"},
                        {"id": "N2", "type": "range_membership", "input": "pred.meas.serum_creatinine.value.range", "value": "scr.221_309", "return_type": "Bool"},
                        {"id": "N3", "type": "range_membership", "input": "pred.meas.egfr.value.range", "value": "egfr.20_30", "return_type": "Bool"},
                        {"id": "ROOT", "type": "combine", "operator": "or", "inputs": ["N1", "N2", "N3"], "return_type": "Bool"},
                    ],
                    "root": "ROOT",
                },
                "action": {"subjects": ["med.class.sglt2i"], "permission": "continue", "intent": "continue_current_therapy", "requirements": ["monitor renal function"]},
                "source_ids": "q2",
            },
            {
                "id": "rule.sglt2i_preop_stop",
                "label": "Stop SGLT2i before planned surgery",
                "input_predicates": ["pred.med.class.sglt2i.on", "pred.proc.surgery.planned"],
                "condition_dag": {
                    "nodes": [
                        {"id": "ROOT", "type": "combine", "operator": "and", "inputs": ["pred.med.class.sglt2i.on", "pred.proc.surgery.planned"], "return_type": "Bool"}
                    ],
                    "root": "ROOT",
                },
                "action": {
                    "subjects": ["med.class.sglt2i"],
                    "permission": "stop",
                    "intent": "temporary_hold",
                    "timing": {"relative_to": "proc.surgery.scheduled_start", "offset": {"value_min": 3, "value_max": 4, "unit": "days", "direction": "before"}},
                },
                "source_ids": "q3",
            },
        ]
    }

    parsed = SubmitSimplifiedRules.model_validate(payload)
    rules = [
        ClinicalRule(
            id=item.id,
            label=item.label,
            input_predicates=item.input_predicates,
            condition_dag=item.condition_dag,
            boolean_root=item.boolean_root,
            action=item.action,
        )
        for item in parsed.rules
    ]
    for rule in rules:
        _assert_dag_parseable(rule)


def test_rule_dag_two_cql_examples_with_count_and_coalesce():
    high_suspicion = ClinicalRule.model_validate({
        "id": "rule.cdi_high_clinical_suspicion",
        "label": "High clinical suspicion for CDI",
        "input_predicates": ["pred.obs.diarrhea.list_24h", "pred.meas.leukocyte.most_recent", "pred.meas.body_temperature.list_24h"],
        "condition_dag": {
            "nodes": [
                {"id": "D1", "type": "aggregate", "operator": "count", "input": "pred.obs.diarrhea.list_24h", "return_type": "Integer"},
                {"id": "D2", "type": "compare", "operator": "ge", "left": "D1", "right": 3, "return_type": "Bool"},
                {"id": "WBC", "type": "compare", "operator": "gt", "left": "pred.meas.leukocyte.most_recent", "right": {"value": 15000, "unit": "{Cells}/L"}, "return_type": "Bool"},
                {"id": "TEMP1", "type": "aggregate", "operator": "max", "input": "pred.meas.body_temperature.list_24h", "return_type": "Quantity"},
                {"id": "TEMP2", "type": "compare", "operator": "ge", "left": "TEMP1", "right": {"value": 38, "unit": "Cel"}, "return_type": "Bool"},
                {"id": "ROOT", "type": "combine", "operator": "and", "inputs": ["D2", "WBC", "TEMP2"], "return_type": "Bool"},
            ],
            "root": "ROOT",
        },
        "action": {"subjects": ["output.cdi.high_clinical_suspicion"], "permission": "recommend", "intent": "derive_state"},
    })

    recent_positive_test = ClinicalRule.model_validate({
        "id": "rule.cdi_test_positive_recent",
        "label": "Most recent CDI test in 28 days is positive",
        "input_predicates": ["pred.obs.cdi_test.list_28d", "pred.valueset.positive_result_indicator"],
        "condition_dag": {
            "nodes": [
                {"id": "DATE", "type": "coalesce", "inputs": ["effectiveDateTime", "effectivePeriod.end", "effectivePeriod.start", "issued"], "return_type": "DateTime"},
                {"id": "MR", "type": "aggregate", "operator": "most_recent", "input": "pred.obs.cdi_test.list_28d", "parameters": {"sort_key": "DATE"}, "return_type": "Observation"},
                {"id": "CONCEPT", "type": "library_function", "library_function": "lib.fhir.concept_in_valueset", "input": "MR", "parameters": {"concept_path": "valueCodeableConcept", "valueset": "Positive result indicator VS"}, "return_type": "Bool"},
                {"id": "ROOT", "type": "predicate_ref", "predicate_ref": "CONCEPT", "return_type": "Bool"},
            ],
            "root": "ROOT",
        },
        "action": {"subjects": ["output.cdi.in_population"], "permission": "recommend", "intent": "derive_state"},
    })

    _assert_dag_parseable(high_suspicion)
    _assert_dag_parseable(recent_positive_test)

    nodes = {node.id: node for node in high_suspicion.condition_dag.nodes}
    assert nodes["D1"].type == "aggregate"
    assert nodes["D1"].operator == "count"
    assert nodes["D2"].type == "compare"
    assert nodes["D2"].left == "D1"


def _minimal_action():
    return {"subjects": ["output.test"], "permission": "recommend"}


def _rule_payload(nodes, root="ROOT", boolean_root=None):
    payload = {
        "id": "rule.validator_fixture",
        "label": "Validator fixture",
        "input_predicates": [],
        "condition_dag": {"nodes": nodes, "root": root},
        "action": _minimal_action(),
    }
    if boolean_root is not None:
        payload["boolean_root"] = boolean_root
    return payload


def test_condition_dag_rejects_missing_root():
    with pytest.raises(ValidationError, match="condition_dag.root 'MISSING'"):
        ClinicalRule.model_validate(_rule_payload(
            [{"id": "N1", "type": "literal", "value": 1, "return_type": "Integer"}],
            root="MISSING",
        ))


def test_condition_dag_rejects_non_bool_root():
    with pytest.raises(ValidationError, match="must return Bool"):
        ClinicalRule.model_validate(_rule_payload(
            [{"id": "ROOT", "type": "literal", "value": 1, "return_type": "Integer"}],
        ))


def test_condition_dag_rejects_duplicate_node_ids():
    with pytest.raises(ValidationError, match="Duplicate DAG node id"):
        ClinicalRule.model_validate(_rule_payload([
            {"id": "ROOT", "type": "literal", "value": True, "return_type": "Bool"},
            {"id": "ROOT", "type": "literal", "value": False, "return_type": "Bool"},
        ]))


def test_dag_node_shape_validators():
    with pytest.raises(ValidationError, match="combine DAG node"):
        ConditionDAG.model_validate({
            "nodes": [{"id": "ROOT", "type": "combine", "operator": "and", "return_type": "Bool"}],
            "root": "ROOT",
        })

    with pytest.raises(ValidationError, match="compare DAG node"):
        ConditionDAG.model_validate({
            "nodes": [{"id": "ROOT", "type": "compare", "operator": "ge", "left": "N1", "return_type": "Bool"}],
            "root": "ROOT",
        })

    with pytest.raises(ValidationError, match=r"aggregate\(count\).*must return Integer"):
        ConditionDAG.model_validate({
            "nodes": [
                {"id": "COUNT", "type": "aggregate", "operator": "count", "input": "pred.obs.events", "return_type": "Quantity"},
                {"id": "ROOT", "type": "compare", "operator": "ge", "left": "COUNT", "right": 1, "return_type": "Bool"},
            ],
            "root": "ROOT",
        })

    dag = ConditionDAG.model_validate({
        "nodes": [
            {"id": "COUNT", "type": "aggregate", "operator": "COUNT", "input": "pred.obs.events", "return_type": "int"},
            {"id": "ROOT", "type": "compare", "operator": "ge", "left": "COUNT", "right": 1, "return_type": "Bool"},
        ],
        "root": "ROOT",
    })
    assert {node.id: node for node in dag.nodes}["COUNT"].return_type == "Integer"


def test_clinical_rule_boolean_root_is_resolved_and_validated():
    rule = ClinicalRule.model_validate(_rule_payload(
        [{"id": "N1", "type": "literal", "value": True, "return_type": "Bool"}],
        root="N1",
    ))
    assert rule.boolean_root == "N1"

    rule_with_alt = ClinicalRule.model_validate(_rule_payload(
        [
            {"id": "ROOT", "type": "literal", "value": True, "return_type": "Bool"},
            {"id": "ALT", "type": "literal", "value": True, "return_type": "Bool"},
        ],
        boolean_root="ALT",
    ))
    assert rule_with_alt.boolean_root == "ALT"

    with pytest.raises(ValidationError, match="boolean_root 'MISSING'"):
        ClinicalRule.model_validate(_rule_payload(
            [{"id": "ROOT", "type": "literal", "value": True, "return_type": "Bool"}],
            boolean_root="MISSING",
        ))


def test_temporal_scope_normalizes_supported_aliases_and_rejects_unknown_modes():
    assert TemporalScope.model_validate({"mode": "most_recent_all_time"}).mode == "all_time"
    assert TemporalScope.model_validate({"mode": "count_in_window"}).mode == "count_in_lookback"

    with pytest.raises(ValidationError, match="Unsupported temporal_scope.mode"):
        TemporalScope.model_validate({"mode": "not_a_standard_mode"})


def test_term_extraction_repairs_serialized_term_lists():
    term_payload = {
        "id": "meas.egfr",
        "name": "Estimated glomerular filtration rate",
        "label": "measures",
        "type": "Quantity",
        "clinical_entity": "meas.egfr",
        "concept": "Estimated glomerular filtration rate",
    }
    med_payload = {
        "id": "med.class.sglt2i",
        "name": "SGLT2 inhibitors",
        "class": None,
        "clinical_entity": "med.class.sglt2i",
        "concept": "SGLT2 inhibitor drug class",
    }

    parsed = TermExtractionResult.model_validate({
        "terms": json.dumps([term_payload]),
        "med_terms": f"[({med_payload!r})]",
    })

    assert parsed.terms[0].id == "meas.egfr"
    assert parsed.med_terms[0].id == "med.class.sglt2i"
