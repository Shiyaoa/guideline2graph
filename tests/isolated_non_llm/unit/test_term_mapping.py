"""
术语映射模块测试
"""
import pytest
import tempfile
from pathlib import Path

from pipeline.term_mapping import TermMappingRegistry, get_registry


class TestTermMappingRegistry:
    """映射注册表测试"""

    def setup_method(self):
        TermMappingRegistry.reset_instance()

    def test_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_register_term(self):
        registry = get_registry()
        registry.register_term(
            semantic_id="meas.egfr",
            concept_id=37393011,
            concept_name="eGFR",
            domain_id="Measurement",
            vocabulary_id="SNOMED",
            match_score=95.0,
            match_type="fuzzy"
        )

        assert registry.term_count == 1
        mapping = registry.get_term_mapping("meas.egfr")
        assert mapping is not None
        assert mapping.concept_id == 37393011

    def test_register_med(self):
        registry = get_registry()
        registry.register_med(
            semantic_id="med.metformin",
            concept_id=1123627,
            concept_name="Metformin",
            domain_id="Drug",
            vocabulary_id="RxNorm",
            match_score=100.0,
            match_type="exact"
        )

        assert registry.med_count == 1
        mapping = registry.get_med_mapping("med.metformin")
        assert mapping is not None
        assert mapping.concept_id == 1123627

    def test_synonym_detection(self):
        registry = get_registry()

        registry.register_term("meas.egfr", 37393011, "eGFR", "Measurement", "SNOMED", 95.0, "fuzzy")
        registry.register_term("meas.estimated_gfr", 37393011, "eGFR", "Measurement", "SNOMED", 90.0, "fuzzy")

        synonyms = registry.get_term_synonyms("meas.egfr")
        assert "meas.estimated_gfr" in synonyms

        synonyms2 = registry.get_term_synonyms("meas.estimated_gfr")
        assert "meas.egfr" in synonyms2

    def test_primary_id_selection(self):
        registry = get_registry()

        registry.register_term("meas.egfr", 37393011, "eGFR", "Measurement", "SNOMED", 95.0, "fuzzy")
        registry.register_term("meas.estimated_gfr", 37393011, "eGFR", "Measurement", "SNOMED", 90.0, "fuzzy")
        registry.register_term("meas.gfr_est", 37393011, "eGFR", "Measurement", "SNOMED", 85.0, "fuzzy")

        primary = registry.get_primary_term_id("meas.estimated_gfr")
        assert primary == "meas.egfr"

        primary2 = registry.get_primary_term_id("meas.gfr_est")
        assert primary2 == "meas.egfr"

    def test_save_and_load(self):
        registry = get_registry()

        registry.register_term("meas.egfr", 37393011, "eGFR", "Measurement", "SNOMED", 95.0, "fuzzy")
        registry.register_med("med.metformin", 1123627, "Metformin", "Drug", "RxNorm", 100.0, "exact")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            registry.save(temp_path)

            TermMappingRegistry.reset_instance()
            registry2 = get_registry()
            registry2.load(temp_path)

            assert registry2.term_count == 1
            assert registry2.med_count == 1
            assert registry2.get_term_mapping("meas.egfr") is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_clear(self):
        registry = get_registry()
        registry.register_term("meas.egfr", 37393011, "eGFR", "Measurement", "SNOMED", 95.0, "fuzzy")

        registry.clear()
        assert registry.term_count == 0
        assert registry.get_term_mapping("meas.egfr") is None


class TestStandardization:
    """标准化模块测试"""

    def test_build_id_replacement_map(self):
        from pipeline.standardize_terms import build_id_replacement_map

        registry = get_registry()
        registry.clear()

        registry.register_term("meas.egfr", 37393011, "eGFR", "Measurement", "SNOMED", 95.0, "fuzzy")
        registry.register_term("meas.estimated_gfr", 37393011, "eGFR", "Measurement", "SNOMED", 90.0, "fuzzy")

        term_replacements, med_replacements = build_id_replacement_map(registry)

        assert term_replacements.get("meas.estimated_gfr") == "meas.egfr"

    def test_merge_terms(self):
        from pipeline.standardize_terms import merge_terms

        terms = [
            {"id": "meas.egfr", "name": "eGFR", "label": "measures"},
            {"id": "meas.estimated_gfr", "name": "Estimated GFR", "label": "measures"},
            {"id": "meas.hba1c", "name": "HbA1c", "label": "measures"},
        ]

        replacements = {"meas.estimated_gfr": "meas.egfr"}
        merged = merge_terms(terms, replacements)

        assert len(merged) == 2

        egfr_term = next(t for t in merged if t["id"] == "meas.egfr")
        assert "aliases" in egfr_term
        assert "meas.estimated_gfr" in egfr_term["aliases"]

        hba1c_term = next(t for t in merged if t["id"] == "meas.hba1c")
        assert "aliases" not in hba1c_term

    def test_update_predicate_references(self):
        from pipeline.standardize_terms import update_predicate_references

        predicates = [
            {
                "id": "pred.meas.estimated_gfr.value.lt.30",
                "entity": "meas.estimated_gfr",
                "dependencies": ["meas.estimated_gfr"],
                "retrieve": {"resource": "Observation", "code_binding": "meas.estimated_gfr"},
                "compare": {"operator": "lt", "value": 30, "unit": "mL/min/1.73m2"},
                "formal_definition": "meas.estimated_gfr < 30"
            }
        ]

        term_replacements = {"meas.estimated_gfr": "meas.egfr"}
        updated = update_predicate_references(predicates, term_replacements, {})

        assert updated[0]["dependencies"] == ["meas.egfr"]
        assert updated[0]["entity"] == "meas.egfr"
        assert updated[0]["retrieve"]["code_binding"] == "meas.egfr"
        assert "meas.egfr" in updated[0]["formal_definition"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
