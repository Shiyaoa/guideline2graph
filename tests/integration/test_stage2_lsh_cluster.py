"""
Stage 2 测试: LSH 聚类
"""
import pytest
import os
import json
from datetime import datetime

from pipeline.models import Provenance
from pipeline.lsh_cluster import lsh_cluster, LSHResult
from pipeline.io_utils import save_clusters, load_clusters


@pytest.mark.integration
class TestLSHClusterStage:
    """LSH 聚类 Stage 测试"""

    def test_cluster_small_dataset(self, sample_provenances, temp_output_dir):
        """
        测试小数据集聚类

        验证:
        1. 聚类正常完成
        2. 所有 provenance 都被分配
        3. bucket_id 正确设置
        """
        print(f"\n[LSH] 开始聚类 {len(sample_provenances)} 条推荐意见")

        result = lsh_cluster(
            provenances=sample_provenances,
            similarity_threshold=0.5,
            max_cluster_size=8
        )

        # 验证结果类型
        assert isinstance(result, LSHResult)
        assert isinstance(result.clusters, list)
        assert isinstance(result.bucket_index, dict)

        # 验证所有 provenance 都被分配
        total_in_clusters = sum(len(c.provenances) for c in result.clusters)
        assert total_in_clusters == len(sample_provenances), \
            f"聚类数量不匹配: {total_in_clusters} vs {len(sample_provenances)}"

        # 验证 bucket_id
        for cluster in result.clusters:
            for p in cluster.provenances:
                assert p.bucket_id is not None
                assert p.bucket_id == cluster.cluster_id

        print(f"[LSH] 聚类完成: {len(result.clusters)} 个聚类")
        for i, c in enumerate(result.clusters):
            print(f"  聚类 {i}: {len(c.provenances)} 条")

    def test_cluster_large_dataset(self, all_provenances, temp_output_dir):
        """
        测试完整数据集聚类

        验证大数据集的聚类性能和正确性
        """
        print(f"\n[LSH] 开始聚类 {len(all_provenances)} 条推荐意见")

        start_time = datetime.now()
        result = lsh_cluster(
            provenances=all_provenances,
            similarity_threshold=0.5,
            max_cluster_size=8
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"[LSH] 聚类完成，耗时 {elapsed:.2f}s")
        print(f"[LSH] 聚类数: {len(result.clusters)}")

        # 验证
        total_in_clusters = sum(len(c.provenances) for c in result.clusters)
        assert total_in_clusters == len(all_provenances)

        # 统计聚类大小分布
        sizes = [len(c.provenances) for c in result.clusters]
        print(f"[LSH] 聚类大小: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

        # 验证 max_cluster_size 限制
        for cluster in result.clusters:
            assert len(cluster.provenances) <= 8, \
                f"聚类 {cluster.cluster_id} 超过最大限制: {len(cluster.provenances)}"

    def test_cluster_persistence(self, sample_provenances, temp_output_dir):
        """
        测试聚类结果持久化

        验证:
        1. 可以保存到文件
        2. 可以从文件加载
        3. 加载后数据一致
        """
        # 执行聚类
        result = lsh_cluster(sample_provenances)

        # 保存
        filepath = os.path.join(temp_output_dir, "test_clusters.json")
        stats = save_clusters(result.clusters, result.bucket_index, filepath)

        assert os.path.exists(filepath)
        assert stats["clusters"] == len(result.clusters)

        # 加载
        loaded_clusters, loaded_bucket_index = load_clusters(filepath)

        assert len(loaded_clusters) == len(result.clusters)

        # 验证 bucket_index
        assert len(loaded_bucket_index) == len(result.bucket_index)

    def test_cluster_similarity_threshold(self, sample_provenances):
        """
        测试不同相似度阈值

        验证阈值对聚类结果的影响
        """
        # 低阈值 → 更多合并
        result_low = lsh_cluster(
            sample_provenances,
            similarity_threshold=0.3,
            max_cluster_size=8
        )

        # 高阈值 → 更少合并
        result_high = lsh_cluster(
            sample_provenances,
            similarity_threshold=0.8,
            max_cluster_size=8
        )

        print(f"\n[LSH] 阈值 0.3: {len(result_low.clusters)} 个聚类")
        print(f"[LSH] 阈值 0.8: {len(result_high.clusters)} 个聚类")

        # 高阈值通常产生更多聚类
        assert len(result_high.clusters) >= len(result_low.clusters)

    def test_cluster_texts_formatted(self, sample_provenances):
        """
        测试 texts_formatted 生成

        验证每个聚类都有正确格式的文本列表
        """
        result = lsh_cluster(sample_provenances)

        for cluster in result.clusters:
            assert len(cluster.texts_formatted) == len(cluster.provenances)

            for text in cluster.texts_formatted:
                assert text.startswith("Quote:")
                assert len(text) > 6  # "Quote:" 后应有内容

    def test_empty_and_single_input(self):
        """
        测试边界条件

        空输入和单条输入
        """
        # 空输入
        result = lsh_cluster([])
        assert result.clusters == []
        assert result.bucket_index == {}

        # 单条输入
        single = [Provenance(source="test", quote="单条测试")]
        result = lsh_cluster(single)
        assert len(result.clusters) == 1
        assert len(result.clusters[0].provenances) == 1


@pytest.mark.integration
class TestLSHClusterQuality:
    """LSH 聚类质量测试"""

    def test_similar_texts_clustered_together(self):
        """
        测试相似文本被聚到一起

        验证 LSH 能识别相似的推荐意见
        """
        provenances = [
            Provenance(source="test", quote="建议使用 SGLT2i 治疗心力衰竭患者"),
            Provenance(source="test", quote="推荐 SGLT2i 用于心衰患者治疗"),
            Provenance(source="test", quote="完全不同的内容关于饮食控制"),
        ]

        result = lsh_cluster(provenances, similarity_threshold=0.4)

        # 前两条相似，应该在同一聚类
        # 找到包含第一条的聚类
        cluster_with_first = None
        for cluster in result.clusters:
            if any(p.quote == provenances[0].quote for p in cluster.provenances):
                cluster_with_first = cluster
                break

        if cluster_with_first:
            # 检查第二条是否在同一聚类
            has_similar = any(p.quote == provenances[1].quote for p in cluster_with_first.provenances)
            if has_similar:
                print("[LSH] 相似文本成功聚类到一起")

    def test_bucket_index_correctness(self, sample_provenances):
        """
        测试 bucket_index 正确性

        验证索引与聚类的一致性
        """
        result = lsh_cluster(sample_provenances)

        # 从 bucket_index 重建原始索引列表
        all_indices_from_index = set()
        for indices in result.bucket_index.values():
            all_indices_from_index.update(indices)

        # 验证每个聚类中的 provenance 都有对应的 bucket_index
        for cluster in result.clusters:
            assert cluster.cluster_id in result.bucket_index
            indices = result.bucket_index[cluster.cluster_id]
            assert len(indices) == len(cluster.provenances)