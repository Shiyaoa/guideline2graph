from pipeline.binding_resolver import resolve_term_binding
from pipeline.term_mapping import TermOMOPMapping


def test_measurement_gets_verified_fhir_paths_and_candidate_omop_code():
    term = {
        "id": "meas.egfr",
        "name": "Estimated glomerular filtration rate",
        "label": "measures",
        "type": "Quantity",
        "value_set_binding": {"type": "Unknown", "name": "eGFR", "confidence": 0.2},
        "normalization_confidence": 0.8,
    }
    mapping = TermOMOPMapping(
        semantic_id="meas.egfr",
        concept_id=3016723,
        concept_name="Glomerular filtration rate",
        domain_id="Measurement",
        vocabulary_id="LOINC",
        match_score=91.0,
        match_type="exact",
    )

    resolved = resolve_term_binding(term, mapping)

    assert resolved["binding_status"] == "candidate"
    assert resolved["verified_binding"]["status"] == "verified"
    assert resolved["verified_binding"]["fhir_binding"]["resource"] == "Observation"
    assert resolved["verified_binding"]["fhir_binding"]["value_path"] == "Observation.valueQuantity"
    assert resolved["candidate_binding"]["status"] == "candidate"
    assert resolved["candidate_binding"]["omop_code_binding"]["binding_status"] == "candidate"
    assert resolved["candidate_binding"]["llm"]["value_set_binding"]["type"] == "Unknown"


def test_condition_resource_binding_does_not_require_omop_code():
    term = {
        "id": "cond.ckd",
        "name": "Chronic kidney disease",
        "label": "conditions",
        "type": "Bool",
    }

    resolved = resolve_term_binding(term)

    assert resolved["binding_status"] == "candidate"
    assert resolved["verified_binding"]["status"] == "verified"
    assert resolved["verified_binding"]["fhir_binding"]["resource"] == "Condition"
    assert "No OMOP-derived candidate terminology code" in resolved["verified_binding"]["unresolved_reasons"][0]
