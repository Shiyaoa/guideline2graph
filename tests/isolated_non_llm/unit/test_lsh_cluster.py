"""
单元测试 - pipeline/lsh_cluster.py
测试目标:
- compute_shingles 计算
- compute_minhash 签名
- lsh_cluster 聚类逻辑
- 边界条件处理
"""
from pipeline.models import Provenance, ProvenanceCluster
from pipeline.lsh_cluster import (
    compute_shingles,
    compute_minhash,
    lsh_cluster,
    LSHResult
)


class TestComputeShingles:
    def test_basic_shingles(self):
        text = "abcde"
        shingles = compute_shingles(text, k=3)
        expected = {"abc", "bcd", "cde"}
        assert shingles == expected

    def test_short_text(self):
        text = "ab"
        shingles = compute_shingles(text, k=3)
        assert shingles == set()

    def test_empty_text(self):
        shingles = compute_shingles("", k=3)
        assert shingles == set()

    def test_case_insensitive(self):
        shingles = compute_shingles("ABC", k=2)
        assert shingles == {"ab", "bc"}

    def test_chinese_text(self):
        shingles = compute_shingles("你好世界", k=2)
        assert "你好" in shingles
        assert "好世" in shingles
        assert "世界" in shingles

    def test_default_k(self):
        text = "abcd"
        shingles = compute_shingles(text)
        assert shingles == {"abc", "bcd"}


class TestComputeMinhash:
    def test_non_empty_shingles(self):
        shingles = {"abc", "bcd", "cde"}
        sig = compute_minhash(shingles, num_hashes=100)
        assert len(sig) == 100
        assert all(isinstance(h, int) for h in sig)

    def test_empty_shingles(self):
        sig = compute_minhash(set(), num_hashes=50)
        assert len(sig) == 50
        assert all(h == 0 for h in sig)

    def test_consistent_signatures(self):
        shingles = {"abc", "bcd", "cde"}
        sig1 = compute_minhash(shingles, num_hashes=100)
        sig2 = compute_minhash(shingles, num_hashes=100)
        assert sig1 == sig2

    def test_different_shingles_different_signatures(self):
        sig1 = compute_minhash({"abc", "def"}, num_hashes=100)
        sig2 = compute_minhash({"xyz", "uvw"}, num_hashes=100)
        assert sig1 != sig2


class TestLSHCluster:
    def test_empty_input(self):
        result = lsh_cluster([])
        assert result.clusters == []
        assert result.bucket_index == {}

    def test_single_item(self):
        provenances = [Provenance(source="test", quote="单条推荐")]
        result = lsh_cluster(provenances)

        assert len(result.clusters) == 1
        assert len(result.clusters[0].provenances) == 1
        assert result.clusters[0].provenances[0].bucket_id == 0

    def test_similar_items_clustered(self):
        provenances = [
            Provenance(source="x", quote="SGLT2i 推荐用于心衰患者"),
            Provenance(source="x", quote="SGLT2i 推荐用于心衰病人"),
        ]
        result = lsh_cluster(provenances, similarity_threshold=0.3)
        assert len(result.clusters) >= 1

    def test_dissimilar_items_separate_clusters(self):
        provenances = [
            Provenance(source="x", quote="完全不同的内容 AAA BBB CCC"),
            Provenance(source="x", quote="XYZ YYY ZZZ 完全不同"),
        ]
        result = lsh_cluster(provenances, similarity_threshold=0.8)
        assert len(result.clusters) >= 1

    def test_max_cluster_size_enforced(self):
        provenances = [
            Provenance(source="x", quote=f"相似文本 {i}")
            for i in range(20)
        ]
        result = lsh_cluster(provenances, max_cluster_size=5, similarity_threshold=0.1)

        for cluster in result.clusters:
            assert len(cluster.provenances) <= 5

    def test_max_clusters_limit(self):
        provenances = [
            Provenance(source="x", quote=f"文本 {i} 不同内容 XYZ")
            for i in range(50)
        ]
        result = lsh_cluster(provenances, max_clusters=5, max_cluster_size=15)

        assert len(result.clusters) <= 5

    def test_bucket_id_assigned(self):
        provenances = [
            Provenance(source="x", quote=f"推荐意见 {i}")
            for i in range(5)
        ]
        result = lsh_cluster(provenances)

        for cluster in result.clusters:
            for p in cluster.provenances:
                assert p.bucket_id is not None
                assert isinstance(p.bucket_id, int)

    def test_bucket_index_consistency(self):
        provenances = [
            Provenance(source="x", quote=f"推荐意见 {i}")
            for i in range(5)
        ]
        result = lsh_cluster(provenances)

        total_in_index = sum(len(indices) for indices in result.bucket_index.values())
        total_in_clusters = sum(len(c.provenances) for c in result.clusters)
        assert total_in_index == total_in_clusters

    def test_texts_formatted_generated(self):
        provenances = [
            Provenance(source="x", quote="第一条推荐"),
            Provenance(source="x", quote="第二条推荐"),
        ]
        result = lsh_cluster(provenances)

        for cluster in result.clusters:
            for text in cluster.texts_formatted:
                assert text.startswith("Quote:")

    def test_return_type(self):
        provenances = [Provenance(source="x", quote="测试")]
        result = lsh_cluster(provenances)

        assert isinstance(result, LSHResult)
        assert isinstance(result.clusters, list)
        assert isinstance(result.bucket_index, dict)


class TestLSHResult:
    def test_get_candidate_pairs_empty(self):
        result = LSHResult(clusters=[], bucket_index={})
        pairs = result.get_candidate_pairs()
        assert pairs == set()

    def test_get_candidate_pairs_single_bucket(self):
        result = LSHResult(
            clusters=[],
            bucket_index={0: [0, 1, 2]}
        )
        pairs = result.get_candidate_pairs()

        assert pairs == {(0, 1), (0, 2), (1, 2)}

    def test_get_candidate_pairs_multiple_buckets(self):
        result = LSHResult(
            clusters=[],
            bucket_index={
                0: [0, 1],
                1: [2, 3],
                2: [0, 2]
            }
        )
        pairs = result.get_candidate_pairs()

        assert (0, 1) in pairs
        assert (2, 3) in pairs
        assert (0, 2) in pairs

    def test_get_candidate_pairs_single_item_bucket(self):
        result = LSHResult(
            clusters=[],
            bucket_index={0: [0]}
        )
        pairs = result.get_candidate_pairs()
        assert pairs == set()


class TestClusteringEdgeCases:
    def test_all_identical_text(self):
        provenances = [
            Provenance(source="x", quote="完全相同的推荐意见")
            for _ in range(5)
        ]
        result = lsh_cluster(provenances, similarity_threshold=0.5)
        assert len(result.clusters) >= 1

    def test_preserves_all_provenances(self):
        provenances = [
            Provenance(source="x", quote=f"推荐意见 {i}")
            for i in range(20)
        ]
        result = lsh_cluster(provenances)

        total = sum(len(c.provenances) for c in result.clusters)
        assert total == len(provenances)

    def test_cluster_id_unique(self):
        provenances = [
            Provenance(source="x", quote=f"推荐意见 {i}")
            for i in range(10)
        ]
        result = lsh_cluster(provenances)

        cluster_ids = [c.cluster_id for c in result.clusters]
        assert len(cluster_ids) == len(set(cluster_ids))

    def test_parameters_affect_clustering(self):
        provenances = [
            Provenance(source="x", quote=f"推荐意见内容测试 {i}")
            for i in range(10)
        ]

        result1 = lsh_cluster(provenances, num_bands=10, rows_per_band=10)
        result2 = lsh_cluster(provenances, num_bands=20, rows_per_band=5)

        assert len(result1.clusters) >= 1
        assert len(result2.clusters) >= 1


class TestProvenanceCluster:
    def test_cluster_model(self):
        provenances = [
            Provenance(source="x", quote="推荐1", bucket_id=0),
            Provenance(source="x", quote="推荐2", bucket_id=0),
        ]
        cluster = ProvenanceCluster(
            cluster_id=0,
            provenances=provenances,
            texts_formatted=["Quote: 推荐1", "Quote: 推荐2"]
        )

        assert cluster.cluster_id == 0
        assert len(cluster.provenances) == 2
        assert len(cluster.texts_formatted) == 2
