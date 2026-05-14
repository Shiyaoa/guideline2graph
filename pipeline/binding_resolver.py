"""Deterministic FHIR resource/path binding and OMOP-derived code candidates.

LLM extraction may provide useful clinical semantics, but it must not be the
authority for executable FHIR or terminology bindings. This module keeps the
stable FHIR resource/path mapping in code and uses OMOP mappings only to produce
auditable candidate code bindings.
"""
from __future__ import annotations

import csv
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import BindingEvidence, BindingResolution, CodeBinding
from .term_mapping import TermMappingRegistry, TermOMOPMapping, get_registry


FHIR_RESOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "measurement": {
        "resource": "Observation",
        "code_path": "Observation.code",
        "value_path": "Observation.valueQuantity",
        "time_path": "Observation.effectiveDateTime",
        "status_path": "Observation.status",
        "status_values": ["final", "amended", "corrected"],
    },
    "observation": {
        "resource": "Observation",
        "code_path": "Observation.code",
        "value_path": "Observation.value[x]",
        "time_path": "Observation.effectiveDateTime",
        "status_path": "Observation.status",
        "status_values": ["final", "amended", "corrected"],
    },
    "condition": {
        "resource": "Condition",
        "code_path": "Condition.code",
        "time_path": "Condition.onsetDateTime",
        "status_path": "Condition.clinicalStatus",
        "verification_status_path": "Condition.verificationStatus",
        "status_values": ["active", "recurrence", "relapse"],
    },
    "procedure": {
        "resource": "Procedure",
        "code_path": "Procedure.code",
        "time_path": "Procedure.performedDateTime",
        "status_path": "Procedure.status",
        "status_values": ["completed", "in-progress"],
    },
    "medication": {
        "resource": "MedicationStatement",
        "code_path": "MedicationStatement.medication[x]",
        "time_path": "MedicationStatement.effectiveDateTime",
        "status_path": "MedicationStatement.status",
        "status_values": ["active", "completed", "intended"],
        "request_resource": "MedicationRequest",
        "request_code_path": "MedicationRequest.medication[x]",
    },
}


OMOP_TABLE_BY_DOMAIN = {
    "Measurement": {"table": "measurement", "concept_role": "measurement_concept_id", "value_path": "value_as_number"},
    "Condition": {"table": "condition_occurrence", "concept_role": "condition_concept_id"},
    "Procedure": {"table": "procedure_occurrence", "concept_role": "procedure_concept_id"},
    "Observation": {"table": "observation", "concept_role": "observation_concept_id"},
    "Drug": {"table": "drug_exposure", "concept_role": "drug_concept_id"},
}


FHIR_SYSTEM_BY_OMOP_VOCAB = {
    "LOINC": "http://loinc.org",
    "SNOMED": "http://snomed.info/sct",
    "SNOMEDCT": "http://snomed.info/sct",
    "ICD10CM": "http://hl7.org/fhir/sid/icd-10-cm",
    "ICD10": "http://hl7.org/fhir/sid/icd-10",
    "ICD9CM": "http://hl7.org/fhir/sid/icd-9-cm",
    "RxNorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "RXNORM": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "ATC": "http://www.whocc.no/atc",
    "CPT4": "http://www.ama-assn.org/go/cpt",
    "HCPCS": "https://www.cms.gov/Medicare/Coding/HCPCSReleaseCodeSets",
    "UCUM": "http://unitsofmeasure.org",
}


def resolve_bindings_for_terms(
    terms: List[dict],
    med_terms: List[dict],
    registry: Optional[TermMappingRegistry] = None,
) -> Tuple[List[dict], List[dict]]:
    """Return term dicts annotated with deterministic/candidate bindings."""
    registry = registry or get_registry()
    resolved_terms = [
        resolve_term_binding(term, registry.get_term_mapping(str(term.get("id", ""))), is_med=False)
        for term in terms
        if isinstance(term, dict)
    ]
    resolved_meds = [
        resolve_term_binding(med, registry.get_med_mapping(str(med.get("id", ""))), is_med=True)
        for med in med_terms
        if isinstance(med, dict)
    ]
    return resolved_terms, resolved_meds


def resolve_term_binding(
    term: dict,
    mapping: Optional[TermOMOPMapping] = None,
    *,
    is_med: bool = False,
) -> dict:
    """Annotate one term dict without mutating the input."""
    resolved = deepcopy(term)
    entity_kind = _infer_entity_kind(resolved, mapping, is_med=is_med)
    fhir_binding = deepcopy(FHIR_RESOURCE_REGISTRY.get(entity_kind, {}))
    omop_binding = _omop_binding_from_mapping(mapping)
    omop_code = _code_binding_from_omop_mapping(mapping)

    candidate_binding: Dict[str, Any] = {
        "status": "candidate",
        "llm": _extract_llm_binding_candidates(resolved),
    }
    candidate_evidence: List[Dict[str, Any]] = [
        BindingEvidence(
            source="llm",
            matched_by="model_suggestion",
            confidence=resolved.get("normalization_confidence"),
            note="LLM-provided binding fields are candidates only.",
        ).model_dump(exclude_none=True)
    ]

    if omop_code:
        candidate_binding["omop_code_binding"] = omop_code.model_dump(exclude_none=True)
        candidate_evidence.append(
            BindingEvidence(
                source="local_omop",
                matched_by=mapping.match_type if mapping else None,
                confidence=_score_to_confidence(mapping.match_score if mapping else None),
                omop_concept_id=mapping.concept_id if mapping else None,
                vocabulary_id=mapping.vocabulary_id if mapping else None,
                note="OMOP-derived code is a terminology candidate until verified against a ValueSet/terminology registry.",
            ).model_dump(exclude_none=True)
        )

    verified_evidence: List[BindingEvidence] = []
    unresolved_reasons: List[str] = []
    if fhir_binding:
        verified_evidence.append(
            BindingEvidence(
                source="fhir_resource_registry",
                matched_by=entity_kind,
                confidence=1.0,
                note="FHIR resource/path mapping comes from the local deterministic registry.",
            )
        )
    else:
        unresolved_reasons.append("No FHIR resource/path registry entry for this term.")

    if not omop_code:
        unresolved_reasons.append("No OMOP-derived candidate terminology code was available.")

    verified_binding = BindingResolution(
        status="verified" if fhir_binding else "unresolved",
        fhir_binding=fhir_binding,
        omop_binding=omop_binding,
        code_bindings=[],
        evidence=verified_evidence,
        unresolved_reasons=unresolved_reasons,
    )

    resolved["candidate_binding"] = candidate_binding
    resolved["verified_binding"] = verified_binding.model_dump(exclude_none=True)
    resolved["binding_evidence"] = candidate_evidence + [
        evidence.model_dump(exclude_none=True) for evidence in verified_evidence
    ]
    resolved["binding_status"] = "candidate" if (fhir_binding or omop_code) else "unresolved"

    if fhir_binding:
        resolved["fhir_binding_hint"] = fhir_binding
        resolved.setdefault("data_bindings", {})
        resolved["data_bindings"].setdefault("FHIR", {})
        resolved["data_bindings"]["FHIR"].update(fhir_binding)
    if omop_binding:
        resolved["omop_binding_hint"] = omop_binding
        resolved.setdefault("data_bindings", {})
        resolved["data_bindings"].setdefault("OMOP", {})
        resolved["data_bindings"]["OMOP"].update(omop_binding)

    return resolved


def get_fhir_resource_registry() -> Dict[str, Dict[str, Any]]:
    """Expose a copy of the deterministic FHIR resource/path registry."""
    return deepcopy(FHIR_RESOURCE_REGISTRY)


def _infer_entity_kind(term: dict, mapping: Optional[TermOMOPMapping], *, is_med: bool) -> str:
    if is_med:
        return "medication"
    if mapping and mapping.domain_id:
        domain_map = {
            "Measurement": "measurement",
            "Condition": "condition",
            "Procedure": "procedure",
            "Observation": "observation",
            "Drug": "medication",
        }
        if mapping.domain_id in domain_map:
            return domain_map[mapping.domain_id]

    label = str(term.get("label") or "").lower()
    value_domain = str(term.get("value_domain") or term.get("type") or "").lower()
    term_id = str(term.get("id") or "").lower()
    if label == "measures" or value_domain == "quantity" or term_id.startswith("meas."):
        return "measurement"
    if label == "conditions" or term_id.startswith("cond."):
        return "condition"
    if label == "procedures" or term_id.startswith("proc."):
        return "procedure"
    if label == "observations" or term_id.startswith("obs."):
        return "observation"
    return "observation"


def _extract_llm_binding_candidates(term: dict) -> Dict[str, Any]:
    candidates = {}
    for key in ("value_set_binding", "code_bindings", "data_bindings", "fhir_binding_hint", "omop_binding_hint"):
        value = term.get(key)
        if value:
            candidates[key] = deepcopy(value)
    return candidates


def _omop_binding_from_mapping(mapping: Optional[TermOMOPMapping]) -> Dict[str, Any]:
    if not mapping:
        return {}
    binding = deepcopy(OMOP_TABLE_BY_DOMAIN.get(mapping.domain_id, {}))
    binding.update(
        {
            "concept_id": mapping.concept_id,
            "concept_name": mapping.concept_name,
            "domain_id": mapping.domain_id,
            "vocabulary_id": mapping.vocabulary_id,
            "match_score": mapping.match_score,
            "match_type": mapping.match_type,
            "binding_status": "candidate",
        }
    )
    concept_meta = _get_omop_concept(mapping.concept_id)
    if concept_meta.get("concept_code"):
        binding["concept_code"] = concept_meta["concept_code"]
    return binding


def _code_binding_from_omop_mapping(mapping: Optional[TermOMOPMapping]) -> Optional[CodeBinding]:
    if not mapping:
        return None
    concept_meta = _get_omop_concept(mapping.concept_id)
    concept_code = concept_meta.get("concept_code")
    system = FHIR_SYSTEM_BY_OMOP_VOCAB.get(mapping.vocabulary_id)
    if not concept_code or not system:
        return CodeBinding(
            type="OMOPConcept",
            system=mapping.vocabulary_id,
            code=mapping.concept_id,
            display=mapping.concept_name,
            confidence=_score_to_confidence(mapping.match_score),
            binding_status="candidate",
            source="local_omop",
            omop_concept_id=mapping.concept_id,
            vocabulary_id=mapping.vocabulary_id,
        )
    return CodeBinding(
        type="Code",
        system=system,
        code=concept_code,
        display=mapping.concept_name,
        confidence=_score_to_confidence(mapping.match_score),
        binding_status="candidate",
        source="local_omop",
        omop_concept_id=mapping.concept_id,
        vocabulary_id=mapping.vocabulary_id,
    )


_concept_row_index: Optional[Dict[str, Dict[str, str]]] = None
_concept_index_lock = threading.Lock()


def _get_omop_concept_row_index() -> Dict[str, Dict[str, str]]:
    """Load OMOP CONCEPT.csv once per process; keyed by concept_id string."""
    global _concept_row_index
    if _concept_row_index is not None:
        return _concept_row_index
    with _concept_index_lock:
        if _concept_row_index is not None:
            return _concept_row_index
        concept_path = Path(__file__).resolve().parents[1] / "omop_normalizer" / "CONCEPT.csv"
        idx: Dict[str, Dict[str, str]] = {}
        if concept_path.exists():
            with concept_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    cid = row.get("concept_id")
                    if cid is not None:
                        idx[str(cid)] = row
        _concept_row_index = idx
        return _concept_row_index


def _get_omop_concept(concept_id: int) -> Dict[str, str]:
    row = _get_omop_concept_row_index().get(str(concept_id))
    return dict(row) if row else {}


def _score_to_confidence(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    return max(0.0, min(float(score) / 100.0, 1.0))
