"""
完整 Pipeline 测试 (E2E)
从 LSH 聚类开始，测试完整的知识抽取流程
使用真实 LLM API（通过 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 配置）
"""
import pytest
import os
import json
from datetime import datetime

from pipeline.models import Provenance
from pipeline.lsh_cluster import lsh_cluster
from pipeline.io_utils import save_clusters, load_provenances, save_provenances


def llm_config_from_env():
    from pipeline.config import LLMConfig

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        pytest.skip("LLM_API_KEY is required for real API integration tests")
    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=os.getenv("LLM_MODEL", "deepseek-v4-pro"),
    )


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """完整 Pipeline 测试"""

    def test_pipeline_from_lsh_small(self, sample_provenances, temp_output_dir):
        """
        小规模 Pipeline 测试

        从 LSH 聚类开始，完成整个知识抽取流程
        """
        from pipeline.graph_api import ClinicalGuidelinePipeline
        from pipeline.config import set_config, PipelineConfig, PathConfig

        # 配置
        config = PipelineConfig(
            llm=llm_config_from_env(),
            paths=PathConfig(gen_dir=temp_output_dir)
        )
        set_config(config)

        print(f"\n[Pipeline] 开始处理 {len(sample_provenances)} 条推荐意见")

        # Step 1: LSH 聚类
        print("\n[Step 1] LSH 聚类...")
        start_time = datetime.now()
        lsh_result = lsh_cluster(sample_provenances, max_cluster_size=5)
        cluster_time = (datetime.now() - start_time).total_seconds()
        print(f"[Step 1] 完成，耗时 {cluster_time:.2f}s，生成 {len(lsh_result.clusters)} 个聚类")

        # 保存聚类结果
        save_clusters(lsh_result.clusters, lsh_result.bucket_index,
                     os.path.join(temp_output_dir, "clusters.json"))

        # Step 2: 创建 Pipeline 并处理
        print("\n[Step 2] 启动知识抽取 Pipeline...")
        pipeline = ClinicalGuidelinePipeline()

        all_terms = []
        all_med_terms = []
        all_predicates = []
        all_rules = []

        extraction_start = datetime.now()

        for cluster in lsh_result.clusters:
            print(f"\n  处理聚类 {cluster.cluster_id} ({len(cluster.provenances)} 条)...")

            from pipeline.graph import process_cluster_node
            from pipeline.models import ClusterState

            state = ClusterState(
                cluster_id=cluster.cluster_id,
                provenances=cluster.provenances,
                texts_formatted=cluster.texts_formatted,
                terms=[],
                med_terms=[],
                predicates=[],
                rules=[]
            )

            result = process_cluster_node(state)

            all_terms.extend(result.get("terms", []))
            all_med_terms.extend(result.get("med_terms", []))
            all_predicates.extend(result.get("predicates", []))
            all_rules.extend(result.get("rules", []))

        extraction_time = (datetime.now() - extraction_start).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()

        # Step 3: 输出统计
        print(f"\n[Step 3] 结果汇总:")
        print(f"  术语: {len(all_terms)}")
        print(f"  药物: {len(all_med_terms)}")
        print(f"  谓词: {len(all_predicates)}")
        print(f"  规则: {len(all_rules)}")
        print(f"\n  聚类耗时: {cluster_time:.2f}s")
        print(f"  抽取耗时: {extraction_time:.2f}s")
        print(f"  总耗时: {total_time:.2f}s")

        # 去重统计
        unique_terms = {t.id for t in all_terms}
        unique_meds = {m.id for m in all_med_terms}
        print(f"\n  唯一术语: {len(unique_terms)}")
        print(f"  唯一药物: {len(unique_meds)}")

        # 保存结果
        results = {
            "terms": [t.model_dump() for t in all_terms],
            "med_terms": [m.model_dump() for m in all_med_terms],
            "predicates": [p.model_dump() for p in all_predicates],
            "rules": [r.model_dump() for r in all_rules]
        }

        output_path = os.path.join(temp_output_dir, "pipeline_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n  结果已保存到: {output_path}")

        # 验证
        assert len(all_terms) > 0 or len(all_med_terms) > 0, \
            "应抽取到至少一些术语或药物"

    def test_pipeline_full_dataset(self, all_provenances, temp_output_dir):
        """
        完整数据集 Pipeline 测试

        处理所有推荐意见
        """
        from pipeline.graph_api import ClinicalGuidelinePipeline
        from pipeline.config import set_config, PipelineConfig, PathConfig
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 配置
        config = PipelineConfig(
            llm=llm_config_from_env(),
            paths=PathConfig(gen_dir=temp_output_dir)
        )
        set_config(config)

        print(f"\n[完整 Pipeline] 处理 {len(all_provenances)} 条推荐意见")

        # Step 1: LSH 聚类
        print("\n[Step 1] LSH 聚类...")
        start_time = datetime.now()
        lsh_result = lsh_cluster(all_provenances, max_cluster_size=8)
        cluster_time = (datetime.now() - start_time).total_seconds()
        print(f"[Step 1] 完成，耗时 {cluster_time:.2f}s")
        print(f"  聚类数: {len(lsh_result.clusters)}")
        print(f"  平均聚类大小: {sum(len(c.provenances) for c in lsh_result.clusters) / len(lsh_result.clusters):.1f}")

        # 保存聚类
        save_clusters(lsh_result.clusters, lsh_result.bucket_index,
                     os.path.join(temp_output_dir, "clusters.json"))

        # Step 2: 并行处理聚类
        print("\n[Step 2] 并行知识抽取...")

        from pipeline.graph import process_cluster_node
        from pipeline.models import ClusterState

        all_terms = []
        all_med_terms = []
        all_predicates = []
        all_rules = []

        extraction_start = datetime.now()
        processed = 0
        failed = 0

        # 使用并行处理
        max_workers = 3  # 限制并发以避免 API 限流

        def process_cluster(cluster):
            try:
                state = ClusterState(
                    cluster_id=cluster.cluster_id,
                    provenances=cluster.provenances,
                    texts_formatted=cluster.texts_formatted,
                    terms=[],
                    med_terms=[],
                    predicates=[],
                    rules=[]
                )
                return process_cluster_node(state)
            except Exception as e:
                print(f"  [ERROR] 聚类 {cluster.cluster_id} 失败: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_cluster, cluster): cluster
                for cluster in lsh_result.clusters
            }

            for future in as_completed(futures):
                cluster = futures[future]
                try:
                    result = future.result(timeout=300)
                    if result:
                        all_terms.extend(result.get("terms", []))
                        all_med_terms.extend(result.get("med_terms", []))
                        all_predicates.extend(result.get("predicates", []))
                        all_rules.extend(result.get("rules", []))
                        processed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  [ERROR] 聚类 {cluster.cluster_id} 超时或失败: {e}")
                    failed += 1

                # 进度显示
                if (processed + failed) % 5 == 0:
                    print(f"  进度: {processed + failed}/{len(lsh_result.clusters)}")

        extraction_time = (datetime.now() - extraction_start).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()

        # Step 3: 结果汇总
        print(f"\n[Step 3] 结果汇总:")
        print(f"  成功: {processed}, 失败: {failed}")
        print(f"  术语: {len(all_terms)}")
        print(f"  药物: {len(all_med_terms)}")
        print(f"  谓词: {len(all_predicates)}")
        print(f"  规则: {len(all_rules)}")
        print(f"\n  聚类耗时: {cluster_time:.2f}s")
        print(f"  抽取耗时: {extraction_time:.2f}s")
        print(f"  总耗时: {total_time:.2f}s")

        # 去重
        unique_term_ids = {t.id for t in all_terms}
        unique_med_ids = {m.id for m in all_med_terms}
        print(f"\n  唯一术语: {len(unique_term_ids)}")
        print(f"  唯一药物: {len(unique_med_ids)}")

        # 保存最终结果
        results = {
            "metadata": {
                "total_provenances": len(all_provenances),
                "total_clusters": len(lsh_result.clusters),
                "processed": processed,
                "failed": failed,
                "cluster_time": cluster_time,
                "extraction_time": extraction_time,
                "total_time": total_time
            },
            "terms": [t.model_dump() for t in all_terms],
            "med_terms": [m.model_dump() for m in all_med_terms],
            "predicates": [p.model_dump() for p in all_predicates],
            "rules": [r.model_dump() for r in all_rules]
        }

        output_path = os.path.join(temp_output_dir, "full_pipeline_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n  结果已保存到: {output_path}")

        # 验证基本质量
        success_rate = processed / len(lsh_result.clusters)
        print(f"\n  成功率: {success_rate:.1%}")

        assert success_rate >= 0.8, f"成功率过低: {success_rate:.1%}"
        assert len(all_terms) > 0 or len(all_med_terms) > 0, \
            "应抽取到至少一些术语或药物"


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineResume:
    """Pipeline 恢复测试"""

    def test_pipeline_resume_from_cache(self, sample_provenances, temp_output_dir):
        """
        测试从缓存恢复

        验证 pipeline 可以从中断处继续
        """
        from pipeline.config import set_config, PipelineConfig, PathConfig
        from pipeline.io_utils import save_cluster_cache, load_cluster_cache

        config = PipelineConfig(
            llm=llm_config_from_env(),
            paths=PathConfig(gen_dir=temp_output_dir)
        )
        set_config(config)

        # 第一次运行：处理部分聚类
        print("\n[恢复测试] 第一次运行...")
        lsh_result = lsh_cluster(sample_provenances, max_cluster_size=5)

        # 模拟部分处理
        cache = {}
        for i, cluster in enumerate(lsh_result.clusters[:2]):
            cache[cluster.cluster_id] = {
                "terms": [],
                "med_terms": [],
                "predicates": [],
                "rules": []
            }

        # 保存缓存
        cache_path = os.path.join(temp_output_dir, "cluster_cache.json")
        save_cluster_cache(cache, cache_path)

        # 加载缓存
        loaded_cache = load_cluster_cache(cache_path)

        print(f"[恢复测试] 缓存中有 {len(loaded_cache)} 个聚类")

        # 验证
        assert len(loaded_cache) == 2

        # 第二次运行：从缓存继续
        print("\n[恢复测试] 第二次运行（从缓存继续）...")

        # 模拟跳过已处理的聚类
        processed_ids = set(loaded_cache.keys())
        remaining = [c for c in lsh_result.clusters if c.cluster_id not in processed_ids]

        print(f"[恢复测试] 跳过 {len(processed_ids)} 个已处理聚类")
        print(f"[恢复测试] 剩余 {len(remaining)} 个聚类待处理")


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineOutput:
    """Pipeline 输出验证测试"""

    def test_output_structure(self, sample_provenances, temp_output_dir):
        """
        测试输出结构

        验证所有输出字段格式正确
        """
        from pipeline.graph import process_cluster_node
        from pipeline.models import ClusterState, Term, MedicationTerm, Predicates, ClinicalRule
        from pipeline.config import set_config, PipelineConfig, PathConfig

        set_config(PipelineConfig(llm=llm_config_from_env(), paths=PathConfig(gen_dir=temp_output_dir)))

        lsh_result = lsh_cluster(sample_provenances[:5], max_cluster_size=5)
        cluster = lsh_result.clusters[0]

        state = ClusterState(
            cluster_id=cluster.cluster_id,
            provenances=cluster.provenances,
            texts_formatted=cluster.texts_formatted,
            terms=[],
            med_terms=[],
            predicates=[],
            rules=[]
        )

        result = process_cluster_node(state)

        # 验证术语结构
        for term in result.get("terms", []):
            assert isinstance(term, Term)
            assert term.id is not None
            assert term.name is not None
            assert term.label is not None

        # 验证药物结构
        for med in result.get("med_terms", []):
            assert isinstance(med, MedicationTerm)
            assert med.id is not None
            assert med.name is not None

        # 验证规则结构
        for rule in result.get("rules", []):
            assert isinstance(rule, ClinicalRule)
            assert rule.id is not None
            assert rule.action is not None
            assert rule.action.permission is not None

        print("\n[输出验证] 所有结构验证通过")

    def test_json_serialization(self, sample_provenances, temp_output_dir):
        """
        测试 JSON 序列化

        验证所有结果可以正确序列化
        """
        from pipeline.graph import process_cluster_node
        from pipeline.models import ClusterState
        from pipeline.config import set_config, PipelineConfig, PathConfig

        set_config(PipelineConfig(llm=llm_config_from_env(), paths=PathConfig(gen_dir=temp_output_dir)))

        lsh_result = lsh_cluster(sample_provenances[:5], max_cluster_size=5)
        cluster = lsh_result.clusters[0]

        state = ClusterState(
            cluster_id=cluster.cluster_id,
            provenances=cluster.provenances,
            texts_formatted=cluster.texts_formatted,
            terms=[],
            med_terms=[],
            predicates=[],
            rules=[]
        )

        result = process_cluster_node(state)

        # 尝试序列化
        output = {
            "terms": [t.model_dump() for t in result.get("terms", [])],
            "med_terms": [m.model_dump() for m in result.get("med_terms", [])],
            "predicates": [p.model_dump() for p in result.get("predicates", [])],
            "rules": [r.model_dump() for r in result.get("rules", [])]
        }

        # 验证可以序列化为 JSON
        json_str = json.dumps(output, ensure_ascii=False, indent=2)
        assert len(json_str) > 0

        # 验证可以反序列化
        loaded = json.loads(json_str)
        assert "terms" in loaded
        assert "med_terms" in loaded

        print(f"\n[序列化验证] JSON 大小: {len(json_str)} 字节")
        print(f"[序列化验证] 验证通过")
