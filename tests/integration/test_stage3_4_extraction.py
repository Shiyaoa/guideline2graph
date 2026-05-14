"""
Stage 3-4 测试: 术语和规则抽取
使用真实 LLM API（通过 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 配置）
"""
import pytest
import os
import json
from datetime import datetime

from pipeline.models import Provenance, ProvenanceCluster
from pipeline.lsh_cluster import lsh_cluster
from pipeline.config import set_config, PipelineConfig, LLMConfig, PathConfig
from pipeline.standard_library import is_v2_library_function_registered


def assert_v2_predicate(predicate):
    assert predicate.id.startswith("pred.")
    assert predicate.input_shape
    assert predicate.reduction.operator
    assert predicate.final_output_type
    assert predicate.temporal_scope.mode
    assert predicate.temporal_scope.mode != "most_recent_all_time"
    assert predicate.entity
    assert predicate.entity_type
    assert predicate.aspect
    for function_id in predicate.library_function:
        assert is_v2_library_function_registered(function_id), f"unregistered library function id: {function_id}"


def assert_v2_rule(rule):
    assert rule.id.startswith("rule.")
    assert rule.input_predicates
    assert rule.condition_dag.root
    nodes_by_id = {node.id: node for node in rule.condition_dag.nodes}
    assert rule.condition_dag.root in nodes_by_id
    assert nodes_by_id[rule.condition_dag.root].return_type == "Bool"
    for node in rule.condition_dag.nodes:
        if node.library_function:
            assert is_v2_library_function_registered(node.library_function), f"unregistered DAG library function id: {node.library_function}"
    assert rule.priority is not None
    assert hasattr(rule, "output_assembly")
    assert rule.condition is None or isinstance(rule.condition, str)


def llm_config_from_env():
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        pytest.skip("LLM_API_KEY is required for real API integration tests")
    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=os.getenv("LLM_MODEL", "deepseek-v4-pro"),
    )


def setup_test_config(temp_output_dir):
    """设置测试配置"""
    config = PipelineConfig(
        llm=llm_config_from_env(),
        paths=PathConfig(gen_dir=temp_output_dir)
    )
    set_config(config)
    return config


@pytest.mark.integration
@pytest.mark.slow
class TestTermExtractionStage:
    """术语抽取 Stage 测试"""

    def test_extract_terms_single_cluster(self, sample_provenances, temp_output_dir):
        """
        测试单个聚类的术语抽取

        使用真实 LLM 抽取术语和药物
        """
        from pipeline.graph_api import extract_terms_stage

        setup_test_config(temp_output_dir)

        # 只取1条推荐意见进行测试，避免LLM响应截断
        test_provenances = sample_provenances[:1]

        # 先进行聚类
        lsh_result = lsh_cluster(test_provenances, max_cluster_size=3)

        if not lsh_result.clusters:
            pytest.skip("没有可用的聚类")

        # 只取第一个聚类测试
        test_clusters = [lsh_result.clusters[0]]
        cluster = test_clusters[0]

        print(f"\n[术语抽取] 测试聚类 {cluster.cluster_id}，包含 {len(cluster.provenances)} 条推荐")

        # 执行术语抽取
        start_time = datetime.now()

        result = extract_terms_stage(
            clusters=test_clusters,
            max_concurrency=1,
            persist_cluster_cache=False
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[术语抽取] 完成，耗时 {elapsed:.2f}s")

        # 验证结果
        terms = result.get("terms", [])
        med_terms = result.get("med_terms", [])

        print(f"[术语抽取] 抽取到 {len(terms)} 个术语, {len(med_terms)} 个药物")

        if terms:
            print("  术语示例:")
            for t in terms[:3]:
                print(f"    - {t.name} ({t.label})")

        if med_terms:
            print("  药物示例:")
            for m in med_terms[:3]:
                print(f"    - {m.name}")

        assert result is not None
        assert terms or med_terms, "terms stage should return v2 terms or medication terms"
        for t in terms:
            assert t.id and t.name and t.clinical_entity is not None
            assert t.data_bindings is not None
            assert t.normalization_confidence is None or 0 <= t.normalization_confidence <= 1
        for m in med_terms:
            assert m.id.startswith("med.")
            assert m.clinical_entity is not None
            assert m.data_bindings is not None

    def test_extract_terms_multiple_clusters(self, sample_provenances, temp_output_dir):
        """
        测试多个聚类的术语抽取

        验证并行处理多个聚类
        """
        from pipeline.graph_api import extract_terms_stage

        setup_test_config(temp_output_dir)

        # 聚类
        lsh_result = lsh_cluster(sample_provenances, max_cluster_size=5)

        if len(lsh_result.clusters) < 2:
            pytest.skip("聚类数量不足")

        # 取前 2 个聚类
        test_clusters = lsh_result.clusters[:2]

        print(f"\n[多聚类测试] 测试 {len(test_clusters)} 个聚类")

        start_time = datetime.now()

        result = extract_terms_stage(
            clusters=test_clusters,
            max_concurrency=2,
            persist_cluster_cache=False
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[多聚类测试] 完成，耗时 {elapsed:.2f}s")

        terms = result.get("terms", [])
        med_terms = result.get("med_terms", [])

        print(f"[多聚类测试] 抽取到 {len(terms)} 个术语, {len(med_terms)} 个药物")


@pytest.mark.integration
@pytest.mark.slow
class TestRuleExtractionStage:
    """规则抽取 Stage 测试"""

    def test_extract_rules_single_cluster(self, sample_provenances, temp_output_dir):
        """
        测试单个聚类的规则抽取

        使用真实 LLM 抽取临床规则
        """
        from pipeline.graph_api import extract_rules_stage

        setup_test_config(temp_output_dir)

        # 先进行聚类 (小规模聚类避免LLM响应截断)
        lsh_result = lsh_cluster(sample_provenances, max_cluster_size=3)

        if not lsh_result.clusters:
            pytest.skip("没有可用的聚类")

        test_clusters = [lsh_result.clusters[0]]
        cluster = test_clusters[0]

        print(f"\n[规则抽取] 测试聚类 {cluster.cluster_id}，包含 {len(cluster.provenances)} 条推荐")

        start_time = datetime.now()

        # 规则抽取需要先有 terms 和 med_terms
        # 先运行术语抽取
        from pipeline.graph_api import extract_terms_stage
        terms_result = extract_terms_stage(
            clusters=test_clusters,
            max_concurrency=1,
            persist_cluster_cache=False
        )

        # 再运行规则抽取
        result = extract_rules_stage(
            clusters=test_clusters,
            max_concurrency=1,
            persist_cluster_cache=False
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[规则抽取] 完成，耗时 {elapsed:.2f}s")

        # 验证结果
        rules = result.get("rules", [])
        predicates = result.get("predicates", [])

        print(f"[规则抽取] 抽取到 {len(rules)} 条规则, {len(predicates)} 个谓词")

        if rules:
            print("  规则示例:")
            for r in rules[:3]:
                print(f"    - {r.label}: {r.action.permission.value} {r.action.subjects}")

        if predicates:
            print("  谓词示例:")
            for p in predicates[:3]:
                print(f"    - {p.name}: {p.input_shape} -> {p.final_output_type} ({p.reduction.operator})")

        assert rules, "rules stage should return at least one v2 rule"
        assert predicates, "rules stage should expose predicates used to build v2 rules"
        for p in predicates:
            assert_v2_predicate(p)
        for r in rules:
            assert_v2_rule(r)


@pytest.mark.integration
@pytest.mark.slow
class TestPredicateExtractionStage:
    """谓词抽取 Stage 测试"""

    def test_extract_predicates_single_cluster(self, sample_provenances, temp_output_dir):
        """
        测试单个聚类的谓词抽取
        """
        from pipeline.graph_api import extract_predicates_stage, extract_terms_stage

        setup_test_config(temp_output_dir)

        lsh_result = lsh_cluster(sample_provenances, max_cluster_size=5)

        if not lsh_result.clusters:
            pytest.skip("没有可用的聚类")

        test_clusters = [lsh_result.clusters[0]]
        cluster = test_clusters[0]

        print(f"\n[谓词抽取] 测试聚类 {cluster.cluster_id}")

        # 先抽取术语
        extract_terms_stage(
            clusters=test_clusters,
            max_concurrency=1,
            persist_cluster_cache=False
        )

        # 再抽取谓词
        start_time = datetime.now()

        result = extract_predicates_stage(
            clusters=test_clusters,
            max_concurrency=1,
            persist_cluster_cache=False
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[谓词抽取] 完成，耗时 {elapsed:.2f}s")

        predicates = result.get("predicates", [])

        print(f"[谓词抽取] 抽取到 {len(predicates)} 个谓词")

        if predicates:
            print("  谓词示例:")
            for p in predicates[:5]:
                print(f"    - {p.name}: {p.input_shape} -> {p.final_output_type} ({p.reduction.operator})")

        assert predicates, "predicate stage should return at least one v2 predicate"
        for p in predicates:
            assert_v2_predicate(p)


@pytest.mark.integration
@pytest.mark.slow
class TestExtractionQuality:
    """抽取质量测试"""

    def test_extraction_pipeline(self, sample_provenances, temp_output_dir):
        """
        测试完整抽取流程：术语 → 谓词 → 规则
        """
        from pipeline.graph_api import (
            extract_terms_stage,
            extract_predicates_stage,
            extract_rules_stage
        )

        setup_test_config(temp_output_dir)

        lsh_result = lsh_cluster(sample_provenances, max_cluster_size=5)

        if not lsh_result.clusters:
            pytest.skip("没有可用的聚类")

        test_clusters = lsh_result.clusters[:2]  # 测试前 2 个聚类

        print(f"\n[抽取流程] 测试 {len(test_clusters)} 个聚类")

        # Step 1: 术语抽取
        print("\n[Step 1] 术语抽取...")
        start_time = datetime.now()
        terms_result = extract_terms_stage(
            clusters=test_clusters,
            max_concurrency=2,
            persist_cluster_cache=False
        )
        print(f"  耗时: {(datetime.now() - start_time).total_seconds():.2f}s")
        print(f"  术语: {len(terms_result.get('terms', []))}")
        print(f"  药物: {len(terms_result.get('med_terms', []))}")

        # Step 2: 谓词抽取
        print("\n[Step 2] 谓词抽取...")
        start_time = datetime.now()
        pred_result = extract_predicates_stage(
            clusters=test_clusters,
            max_concurrency=2,
            persist_cluster_cache=False
        )
        print(f"  耗时: {(datetime.now() - start_time).total_seconds():.2f}s")
        print(f"  谓词: {len(pred_result.get('predicates', []))}")

        # Step 3: 规则抽取
        print("\n[Step 3] 规则抽取...")
        start_time = datetime.now()
        rules_result = extract_rules_stage(
            clusters=test_clusters,
            max_concurrency=2,
            persist_cluster_cache=False
        )
        print(f"  耗时: {(datetime.now() - start_time).total_seconds():.2f}s")
        print(f"  规则: {len(rules_result.get('rules', []))}")

        # 验证抽取质量
        total_terms = len(terms_result.get('terms', []))
        total_meds = len(terms_result.get('med_terms', []))
        total_predicates = len(pred_result.get('predicates', []))
        total_rules = len(rules_result.get('rules', []))

        print(f"\n[抽取结果汇总]")
        print(f"  术语: {total_terms}")
        print(f"  药物: {total_meds}")
        print(f"  谓词: {total_predicates}")
        print(f"  规则: {total_rules}")

        # 基本质量断言
        assert total_terms > 0 or total_meds > 0, "应抽取到至少一些术语或药物"
