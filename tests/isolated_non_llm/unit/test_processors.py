"""
单元测试 - pipeline/processors.py
测试目标:
- TermProcessor 匹配逻辑
- MedicationProcessor 匹配逻辑
- 缓存命中/未命中
- 阈值过滤
"""
import pytest
from unittest.mock import Mock, patch

from pipeline.models import (
    Term, MedicationTerm, TermLabel,
    TermList, MedicationTermList, TermExtractionResult
)
from pipeline.processors import (
    TermProcessor,
    MedicationProcessor,
    CombinedTermProcessor,
    PredicateProcessor,
    MatchResult,
    process_terms,
    process_med_terms,
    process_combined_terms,
    process_predicates,
)


class TestMatchResult:
    def test_create_matched_result(self):
        result = MatchResult(
            matched=True,
            concept_id=12345,
            concept_name="EGFR",
            domain_id="Measurement",
            vocabulary_id="LOINC",
            match_score=95.0,
            match_type="exact"
        )

        assert result.matched is True
        assert result.concept_id == 12345
        assert result.concept_name == "EGFR"

    def test_create_unmatched_result(self):
        result = MatchResult(matched=False)

        assert result.matched is False
        assert result.concept_id == 0
        assert result.concept_name == ""


class TestTermProcessor:
    def test_process_with_match(self, mock_omop_matcher, mock_config):
        matcher, cache = mock_omop_matcher

        processor = TermProcessor(enable_cache=True)
        terms = TermList(items=[
            Term(id="t1", name="eGFR", label=TermLabel.MEASURES, type="lab")
        ])

        result = processor.process(terms)

        assert len(result.items) == 1
        matcher.match.assert_called()

    def test_process_without_match(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            matcher = Mock()
            cache = Mock()

            matcher.match = Mock(return_value=[])
            cache.get = Mock(return_value=None)
            mock_get.return_value = (matcher, cache)

            processor = TermProcessor(enable_cache=True)
            terms = TermList(items=[
                Term(id="t1", name="未知术语", label=TermLabel.MEASURES, type="lab")
            ])

            result = processor.process(terms)

            assert len(result.items) == 1
            assert result.items[0].name == "未知术语"

    def test_process_with_cache_hit(self, mock_omop_cached, mock_config):
        matcher, cache = mock_omop_cached

        processor = TermProcessor(enable_cache=True)
        terms = TermList(items=[
            Term(id="t1", name="eGFR", label=TermLabel.MEASURES, type="lab")
        ])

        result = processor.process(terms)

        cache.get.assert_called()
        matcher.match.assert_not_called()

    def test_process_below_threshold(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            matcher = Mock()
            cache = Mock()

            low_score_match = Mock()
            low_score_match.concept_id = 12345
            low_score_match.concept_name = "EGFR"
            low_score_match.match_score = 50.0
            low_score_match.domain_id = "Measurement"
            low_score_match.vocabulary_id = "LOINC"
            low_score_match.match_type = "fuzzy"

            matcher.match = Mock(return_value=[low_score_match])
            cache.get = Mock(return_value=None)
            mock_get.return_value = (matcher, cache)

            processor = TermProcessor(enable_cache=True)
            terms = TermList(items=[
                Term(id="t1", name="xxx", label=TermLabel.MEASURES, type="lab")
            ])

            result = processor.process(terms)

            assert result.items[0].name == "xxx"

    def test_domain_mapping(self, mock_omop_matcher, mock_config):
        matcher, cache = mock_omop_matcher

        processor = TermProcessor(enable_cache=True)
        processor._match_term("eGFR", "measures")
        matcher.match.assert_called_with("eGFR", domain_id="Measurement", top_k=1)

        matcher.match.reset_mock()
        processor._match_term("心衰", "conditions")
        # _preprocess_name_for_omop 将「心衰」映射为 heart failure
        matcher.match.assert_called_with("heart failure", domain_id="Condition", top_k=1)

    def test_create_term_from_match(self):
        match = MatchResult(
            matched=True,
            concept_id=12345,
            concept_name="EGFR",
            domain_id="Measurement",
            vocabulary_id="LOINC",
            match_score=95.0,
            match_type="exact"
        )

        original = Term(id="old", name="eGFR", label=TermLabel.MEASURES, type="lab", unit="ml/min")
        result = TermProcessor._create_term(match, original)

        assert result.id == "12345"
        assert result.name == "EGFR"
        assert result.label == TermLabel.MEASURES
        assert result.unit == "ml/min"


class TestMedicationProcessor:
    def test_process_medication(self, mock_omop_matcher, mock_config):
        matcher, cache = mock_omop_matcher

        processor = MedicationProcessor(enable_cache=True)
        meds = MedicationTermList(items=[
            MedicationTerm(id="med.1", name="SGLT2i", drug_class="med.sglt2i")
        ])

        result = processor.process(meds)

        assert len(result.items) == 1
        matcher.match.assert_called()

    def test_drug_domain_search(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            matcher = Mock()
            cache = Mock()

            matcher.match = Mock(return_value=[])
            cache.get = Mock(return_value=None)
            mock_get.return_value = (matcher, cache)

            processor = MedicationProcessor(enable_cache=True)
            processor._match_med("SGLT2i")

            matcher.match.assert_called_with("SGLT2i", domain_id="Drug", top_k=1)

    def test_medication_cache(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            matcher = Mock()
            cache = Mock()

            cache.get = Mock(return_value={
                "concept_id": 12345,
                "concept_name": "Dapagliflozin",
                "domain_id": "Drug",
                "vocabulary_id": "RxNorm",
                "match_score": 100.0
            })
            mock_get.return_value = (matcher, cache)

            processor = MedicationProcessor(enable_cache=True)
            result = processor._match_med("达格列净")

            assert result.matched is True
            assert result.concept_name == "Dapagliflozin"
            matcher.match.assert_not_called()


class TestCombinedTermProcessor:
    def test_process_combined(self, mock_omop_matcher, mock_config):
        matcher, cache = mock_omop_matcher

        processor = CombinedTermProcessor(enable_cache=True)
        result = TermExtractionResult(
            terms=[Term(id="t1", name="eGFR", label=TermLabel.MEASURES, type="lab")],
            med_terms=[MedicationTerm(id="m1", name="SGLT2i", drug_class=None)]
        )

        processed = processor.process(result)

        assert len(processed.terms) == 1
        assert len(processed.med_terms) == 1

    def test_cross_domain_search(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            matcher = Mock()
            cache = Mock()

            match_result = Mock()
            match_result.concept_id = 12345
            match_result.concept_name = "EGFR"
            match_result.match_score = 95.0
            match_result.domain_id = "Measurement"
            match_result.vocabulary_id = "LOINC"
            match_result.match_type = "exact"

            matcher.match = Mock(return_value=[match_result])
            cache.get = Mock(return_value=None)
            mock_get.return_value = (matcher, cache)

            processor = CombinedTermProcessor(enable_cache=True)
            result = processor._match_in_omop(
                "eGFR",
                domain_hints=["Measurement", "Condition"]
            )

            assert result.matched is True
            matcher.match.assert_called()


class TestPredicateProcessor:
    def test_process_predicates(self, sample_predicates, mock_config):
        from pipeline.models import PredicatesList

        processor = PredicateProcessor()
        preds = PredicatesList(items=sample_predicates)

        result = processor.process(preds)

        assert len(result.items) == len(sample_predicates)


class TestConvenienceFunctions:
    def test_process_terms_function(self, mock_omop_matcher, mock_config):
        terms = TermList(items=[
            Term(id="t1", name="eGFR", label=TermLabel.MEASURES, type="lab")
        ])

        result = process_terms(terms)
        assert len(result.items) == 1

    def test_process_med_terms_function(self, mock_omop_matcher, mock_config):
        meds = MedicationTermList(items=[
            MedicationTerm(id="m1", name="SGLT2i", drug_class=None)
        ])

        result = process_med_terms(meds)
        assert len(result.items) == 1

    def test_process_combined_terms_function(self, mock_omop_matcher, mock_config):
        result = TermExtractionResult(
            terms=[Term(id="t1", name="eGFR", label=TermLabel.MEASURES, type="lab")],
            med_terms=[]
        )

        processed = process_combined_terms(result)
        assert len(processed.terms) == 1


class TestCacheBehavior:
    def test_cache_save_on_match(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            matcher = Mock()
            cache = Mock()

            match_result = Mock()
            match_result.concept_id = 12345
            match_result.concept_name = "EGFR"
            match_result.match_score = 95.0
            match_result.domain_id = "Measurement"
            match_result.vocabulary_id = "LOINC"
            match_result.match_type = "exact"

            matcher.match = Mock(return_value=[match_result])
            cache.get = Mock(return_value=None)
            mock_get.return_value = (matcher, cache)

            processor = TermProcessor(enable_cache=True)
            processor._match_term("eGFR", "measures")

            cache.save.assert_called_once()

    def test_cache_disabled(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            matcher = Mock()
            cache = Mock()

            match_result = Mock()
            match_result.concept_id = 12345
            match_result.concept_name = "EGFR"
            match_result.match_score = 95.0
            match_result.domain_id = "Measurement"
            match_result.vocabulary_id = "LOINC"
            match_result.match_type = "exact"

            matcher.match = Mock(return_value=[match_result])
            mock_get.return_value = (matcher, cache)

            processor = TermProcessor(enable_cache=False)
            processor._match_term("eGFR", "measures")

            cache.get.assert_not_called()
            cache.save.assert_not_called()


class TestThresholdFiltering:
    def test_term_threshold(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            from pipeline.config import MatchConfig

            config = MatchConfig(term_threshold=90.0)

            matcher = Mock()
            cache = Mock()

            match_result = Mock()
            match_result.concept_id = 12345
            match_result.concept_name = "EGFR"
            match_result.match_score = 85.0
            match_result.domain_id = "Measurement"
            match_result.vocabulary_id = "LOINC"
            match_result.match_type = "fuzzy"

            matcher.match = Mock(return_value=[match_result])
            cache.get = Mock(return_value=None)
            mock_get.return_value = (matcher, cache)

            processor = TermProcessor(config=config, enable_cache=True)
            result = processor._match_term("eGFR", "measures")

            assert result.matched is False

    def test_med_threshold(self, mock_config):
        with patch('pipeline.processors._get_omop_components') as mock_get:
            from pipeline.config import MatchConfig

            config = MatchConfig(med_threshold=95.0)

            matcher = Mock()
            cache = Mock()

            match_result = Mock()
            match_result.concept_id = 12345
            match_result.concept_name = "Dapagliflozin"
            match_result.match_score = 90.0
            match_result.domain_id = "Drug"
            match_result.vocabulary_id = "RxNorm"
            match_result.match_type = "fuzzy"

            matcher.match = Mock(return_value=[match_result])
            cache.get = Mock(return_value=None)
            mock_get.return_value = (matcher, cache)

            processor = MedicationProcessor(config=config, enable_cache=True)
            result = processor._match_med("达格列净")

            assert result.matched is False
