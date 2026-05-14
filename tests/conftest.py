"""
pytest 共享 fixtures - 用于单元测试和集成测试
"""
import os
import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile

# 添加项目根目录到 Python 路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.models import (
    Term, MedicationTerm, Predicates, ClinicalRule, Provenance,
    ProvenanceCluster, TermLabel, Permission, Action,
    TermList, MedicationTermList, PredicatesList, TermExtractionResult
)


# ============ 临时目录 Fixtures ============

@pytest.fixture
def temp_dir():
    """创建临时目录用于测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_gen_dir(temp_dir):
    """创建临时 gen 目录"""
    gen_dir = Path(temp_dir) / "gen"
    gen_dir.mkdir(exist_ok=True)
    return str(gen_dir)


# ============ Sample Data Fixtures ============

@pytest.fixture
def sample_provenances():
    """样本推荐意见"""
    return [
        Provenance(source="test", quote="建议使用 SGLT2i 治疗心衰患者"),
        Provenance(source="test", quote="肾功能 eGFR < 30 时慎用 SGLT2i"),
        Provenance(source="test", quote="SGLT2i 可降低心血管死亡风险"),
    ]


@pytest.fixture
def sample_terms():
    """样本术语列表"""
    return [
        Term(id="term.1", name="eGFR", label=TermLabel.MEASURES, type="lab", unit="ml/min/1.73m2"),
        Term(id="term.2", name="HbA1c", label=TermLabel.MEASURES, type="lab", unit="%"),
        Term(id="term.3", name="心力衰竭", label=TermLabel.CONDITIONS, type="diagnosis"),
    ]


@pytest.fixture
def sample_med_terms():
    """样本药物术语列表"""
    return [
        MedicationTerm(id="med.sglt2i", name="SGLT2i", drug_class="med.sglt2_inhibitor"),
        MedicationTerm(id="med.dapagliflozin", name="达格列净", drug_class="med.sglt2i"),
    ]


@pytest.fixture
def sample_predicates():
    """样本谓词列表"""
    return [
        Predicates(
            id="pred.meas.egfr.value.lt.30",
            name="eGFR < 30",
            description="Most recent eGFR is below 30.",
            source_text="肾功能 eGFR < 30",
            entity="meas.egfr",
            entity_type="observation",
            aspect="quantity",
            input_shape="List<Observation>",
            reduction={"operator": "most_recent", "output_type": "Quantity"},
            return_type="Bool",
            final_output_type="Bool",
            temporal_scope={"mode": "all_time"},
            data_binding={"FHIR": {"resource": "Observation"}, "OMOP": {"table": "measurement"}},
            library_function=["lib.fhir.most_recent"],
            unit="ml/min/1.73m2",
            quantity_semantics={"unit": "ml/min/1.73m2", "operator": "lt", "value": 30},
            retrieve={"resource": "Observation", "code_binding": "term.1"},
            extract={"path": "valueQuantity", "type": "Quantity"},
            compare={"operator": "lt", "value": 30, "unit": "ml/min/1.73m2"},
            evidence=[{"source_text": "肾功能 eGFR < 30"}],
            dependencies=["term.1"],
        ),
        Predicates(
            id="pred.cond.hf.exists",
            name="Heart failure exists",
            description="Patient has heart failure.",
            source_text="心衰患者",
            entity="term.3",
            entity_type="condition",
            aspect="existence",
            input_shape="List<Condition>",
            reduction={"operator": "exists", "output_type": "Bool"},
            return_type="Bool",
            final_output_type="Bool",
            temporal_scope={"mode": "all_time"},
            data_binding={"FHIR": {"resource": "Condition"}, "OMOP": {"table": "condition_occurrence"}},
            retrieve={"resource": "Condition", "code_binding": "term.3"},
            evidence=[{"source_text": "心衰患者"}],
            dependencies=["term.3"],
        ),
    ]


@pytest.fixture
def sample_rules(sample_med_terms, sample_predicates, sample_provenances):
    """样本临床规则"""
    return [
        ClinicalRule(
            id="rule.1",
            label="心衰患者推荐SGLT2i",
            input_predicates=["pred.cond.hf.exists"],
            condition_dag={
                "nodes": [
                    {"id": "ROOT", "type": "predicate_ref", "predicate_ref": "pred.cond.hf.exists", "return_type": "Bool"}
                ],
                "root": "ROOT"
            },
            boolean_root="ROOT",
            action=Action(
                subjects=["med.sglt2i"],
                permission=Permission.RECOMMEND,
                requirements=[]
            ),
            provenance=[sample_provenances[0]]
        ),
        ClinicalRule(
            id="rule.2",
            label="eGFR<30时慎用SGLT2i",
            input_predicates=["pred.meas.egfr.value.lt.30"],
            condition_dag={
                "nodes": [
                    {"id": "ROOT", "type": "predicate_ref", "predicate_ref": "pred.meas.egfr.value.lt.30", "return_type": "Bool"}
                ],
                "root": "ROOT"
            },
            boolean_root="ROOT",
            action=Action(
                subjects=["med.sglt2i"],
                permission=Permission.CAUTION,
                requirements=["监测肾功能"]
            ),
            provenance=[sample_provenances[1]]
        ),
    ]


@pytest.fixture
def sample_clusters(sample_provenances):
    """样本聚类"""
    return [
        ProvenanceCluster(
            cluster_id=0,
            provenances=sample_provenances[:2],
            texts_formatted=[f"Quote: {p.quote}" for p in sample_provenances[:2]]
        ),
        ProvenanceCluster(
            cluster_id=1,
            provenances=[sample_provenances[2]],
            texts_formatted=[f"Quote: {sample_provenances[2].quote}"]
        ),
    ]


# ============ LLM Mock Fixtures ============

@pytest.fixture
def mock_llm_json_response():
    """模拟 LLM JSON 响应的工厂函数"""
    def _make_response(data: dict):
        mock_msg = Mock()
        mock_msg.content = json.dumps(data, ensure_ascii=False)
        return mock_msg
    return _make_response


@pytest.fixture
def mock_chat_openai():
    """模拟 ChatOpenAI 客户端"""
    with patch('langchain_openai.ChatOpenAI') as mock:
        instance = mock.return_value
        instance.invoke = MagicMock()
        yield instance


@pytest.fixture
def mock_llm_provenance_response(mock_llm_json_response, sample_provenances):
    """模拟 provenance 抽取响应"""
    data = {
        "items": [{"source": p.source, "quote": p.quote} for p in sample_provenances]
    }
    return mock_llm_json_response(data)


@pytest.fixture
def mock_llm_terms_response(mock_llm_json_response, sample_terms, sample_med_terms):
    """模拟术语抽取响应"""
    data = {
        "terms": [t.model_dump() for t in sample_terms],
        "med_terms": [m.model_dump() for m in sample_med_terms]
    }
    return mock_llm_json_response(data)


# ============ OMOP Mock Fixtures ============

@pytest.fixture
def mock_omop_match():
    """创建单个 OMOP 匹配结果"""
    def _create_match(concept_id: int, concept_name: str, match_score: float = 95.0, domain_id: str = "Measurement"):
        mock_result = Mock()
        mock_result.concept_id = concept_id
        mock_result.concept_name = concept_name
        mock_result.match_score = match_score
        mock_result.domain_id = domain_id
        mock_result.vocabulary_id = "LOINC"
        mock_result.match_type = "exact"
        return mock_result
    return _create_match


@pytest.fixture
def mock_omop_matcher(mock_omop_match):
    """模拟 OMOP 匹配器"""
    with patch('pipeline.processors._get_omop_components') as mock_get:
        matcher = Mock()
        cache = Mock()

        # 默认返回一个匹配结果
        matcher.match = Mock(return_value=[
            mock_omop_match(concept_id=12345, concept_name="EGFR", match_score=95.0)
        ])
        cache.get = Mock(return_value=None)
        cache.save = Mock()

        mock_get.return_value = (matcher, cache)
        yield matcher, cache


@pytest.fixture
def mock_omop_cached():
    """模拟已缓存的 OMOP 匹配结果"""
    with patch('pipeline.processors._get_omop_components') as mock_get:
        matcher = Mock()
        cache = Mock()

        # 模拟缓存命中
        cache.get = Mock(return_value={
            "concept_id": 12345,
            "concept_name": "EGFR",
            "domain_id": "Measurement",
            "vocabulary_id": "LOINC",
            "match_score": 100.0,
            "match_type": "cached"
        })
        cache.save = Mock()

        mock_get.return_value = (matcher, cache)
        yield matcher, cache


# ============ Configuration Fixtures ============

@pytest.fixture
def mock_config(temp_gen_dir):
    """模拟配置，使用临时目录"""
    with patch('pipeline.config.get_config') as mock_get:
        from pipeline.config import PipelineConfig, LLMConfig, PathConfig, MatchConfig

        config = PipelineConfig(
            llm=LLMConfig(api_key="test-key"),
            paths=PathConfig(gen_dir=temp_gen_dir),
            match=MatchConfig()
        )
        mock_get.return_value = config
        yield config


@pytest.fixture
def mock_pipeline_config(temp_gen_dir):
    """设置全局配置为临时目录"""
    from pipeline.config import set_config, PipelineConfig, LLMConfig, PathConfig, MatchConfig

    original_config = None
    try:
        from pipeline.config import get_config
        original_config = get_config()
    except:
        pass

    config = PipelineConfig(
        llm=LLMConfig(api_key="test-key"),
        paths=PathConfig(gen_dir=temp_gen_dir),
        match=MatchConfig()
    )
    set_config(config)

    yield config

    # 恢复原始配置
    if original_config:
        set_config(original_config)


# ============ Helper Functions ============

@pytest.fixture
def assert_json_serializable():
    """断言对象可 JSON 序列化"""
    def _assert(obj):
        try:
            json.dumps(obj, ensure_ascii=False, default=str)
            return True
        except (TypeError, ValueError) as e:
            raise AssertionError(f"Object not JSON serializable: {e}")
    return _assert
