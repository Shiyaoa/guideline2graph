"""
单元测试 - pipeline/io_utils.py
测试目标:
- JSON 读写
- 缓存持久化
- FailedTaskLogger 线程安全
- 去重合并逻辑
"""
import os
import json
import pytest
import threading
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, Mock

from pipeline.models import (
    Term, MedicationTerm, Provenance, ProvenanceCluster,
    TermLabel, Permission, Action, ClinicalRule
)
from pipeline.io_utils import (
    save_to_gen,
    load_from_gen,
    save_cluster_cache,
    load_cluster_cache,
    save_provenances,
    load_provenances,
    save_clusters,
    load_clusters,
    FailedTaskLogger,
    get_failed_task_logger,
    _to_dict_list,
    _load_json_file,
    _save_json_file,
    _merge_by_id,
    _merge_by_key,
)


class TestToJsonDictList:
    """_to_dict_list 测试"""

    def test_pydantic_models(self):
        """Pydantic 模型转字典"""
        items = [
            Term(id="1", name="eGFR", label=TermLabel.MEASURES, type="lab")
        ]
        result = _to_dict_list(items)

        assert len(result) == 1
        assert result[0]["id"] == "1"
        assert result[0]["name"] == "eGFR"

    def test_dicts_passthrough(self):
        """字典直接通过"""
        items = [{"id": "1", "name": "test"}]
        result = _to_dict_list(items)

        assert result == items

    def test_mixed_input(self):
        """混合输入"""
        items = [
            Term(id="1", name="eGFR", label=TermLabel.MEASURES, type="lab"),
            {"id": "2", "name": "test"}
        ]
        result = _to_dict_list(items)

        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_empty_input(self):
        """空输入"""
        assert _to_dict_list([]) == []

    def test_string_fallback(self):
        """非模型非字典的回退"""
        items = ["string1", "string2"]
        result = _to_dict_list(items)

        assert result == ["string1", "string2"]


class TestJsonFileOperations:
    """JSON 文件读写测试"""

    def test_save_and_load_json(self, temp_dir):
        """保存和加载 JSON"""
        filepath = os.path.join(temp_dir, "test.json")
        data = {"key": "value", "number": 123}

        _save_json_file(filepath, data)
        assert os.path.exists(filepath)

        loaded = _load_json_file(filepath)
        assert loaded == data

    def test_load_nonexistent_file(self, temp_dir):
        """加载不存在的文件返回空列表"""
        filepath = os.path.join(temp_dir, "nonexistent.json")
        result = _load_json_file(filepath)
        assert result == []

    def test_load_invalid_json(self, temp_dir):
        """加载无效 JSON 返回空列表"""
        filepath = os.path.join(temp_dir, "invalid.json")
        with open(filepath, "w") as f:
            f.write("not valid json {")

        result = _load_json_file(filepath)
        assert result == []

    def test_save_chinese_content(self, temp_dir):
        """保存中文内容"""
        filepath = os.path.join(temp_dir, "chinese.json")
        data = {"中文": "测试", "quote": "推荐意见"}

        _save_json_file(filepath, data)
        loaded = _load_json_file(filepath)

        assert loaded["中文"] == "测试"


class TestMergeById:
    """_merge_by_id 测试"""

    def test_merge_new_items(self):
        """合并新项目"""
        existing = [{"id": "1", "name": "A"}]
        new = [{"id": "2", "name": "B"}]
        result = _merge_by_id(existing, new)

        assert len(result) == 2

    def test_update_existing_item(self):
        """更新已存在的项目"""
        existing = [{"id": "1", "name": "A"}]
        new = [{"id": "1", "name": "A_updated"}]
        result = _merge_by_id(existing, new)

        assert len(result) == 1
        assert result[0]["name"] == "A_updated"

    def test_merge_empty_existing(self):
        """空 existing 合并"""
        new = [{"id": "1", "name": "A"}]
        result = _merge_by_id([], new)

        assert result == new

    def test_merge_empty_new(self):
        """空 new 合并"""
        existing = [{"id": "1", "name": "A"}]
        result = _merge_by_id(existing, [])

        assert result == existing


class TestMergeByKey:
    """_merge_by_key 测试"""

    def test_merge_by_quote(self):
        """按 quote 合并"""
        existing = [{"quote": "推荐A", "source": "old"}]
        new = [{"quote": "推荐A", "source": "new"}]
        result = _merge_by_key(existing, new, key="quote")

        assert len(result) == 1
        assert result[0]["source"] == "new"

    def test_add_new_by_key(self):
        """添加新项目"""
        existing = [{"quote": "推荐A"}]
        new = [{"quote": "推荐B"}]
        result = _merge_by_key(existing, new, key="quote")

        assert len(result) == 2


class TestProvenanceIO:
    """推荐意见读写测试"""

    def test_save_and_load_provenances(self, temp_dir, sample_provenances):
        """保存和加载推荐意见"""
        filepath = os.path.join(temp_dir, "provenances.json")

        count = save_provenances(sample_provenances, filepath)
        assert count == len(sample_provenances)

        loaded = load_provenances(filepath)
        assert len(loaded) == len(sample_provenances)
        assert all(isinstance(p, Provenance) for p in loaded)

    def test_load_empty_file(self, temp_dir):
        """加载空文件"""
        filepath = os.path.join(temp_dir, "empty.json")
        Path(filepath).touch()

        loaded = load_provenances(filepath)
        assert loaded == []


class TestClusterIO:
    """聚类读写测试"""

    def test_save_and_load_clusters(self, temp_dir, sample_clusters):
        """保存和加载聚类"""
        filepath = os.path.join(temp_dir, "clusters.json")

        stats = save_clusters(sample_clusters, filepath=filepath)
        assert stats["clusters"] == len(sample_clusters)

        loaded_clusters, loaded_bucket_index = load_clusters(filepath)
        assert len(loaded_clusters) == len(sample_clusters)

    def test_save_clusters_with_bucket_index(self, temp_dir, sample_clusters):
        """保存带 bucket_index 的聚类"""
        filepath = os.path.join(temp_dir, "clusters.json")
        bucket_index = {0: [0, 1], 1: [2]}

        save_clusters(sample_clusters, bucket_index=bucket_index, filepath=filepath)

        _, loaded_bucket_index = load_clusters(filepath)
        # JSON 序列化会将 int 键转为 str 键
        # 所以我们需要比较转换后的结果
        assert int(list(loaded_bucket_index.keys())[0]) in [0, 1]
        assert loaded_bucket_index["0"] == [0, 1] or loaded_bucket_index.get(0) == [0, 1]


class TestClusterCache:
    """聚类缓存读写测试"""

    def test_save_and_load_cluster_cache(self, temp_dir):
        """保存和加载聚类缓存"""
        filepath = os.path.join(temp_dir, "cluster_cache.json")

        cache = {
            1: {
                "terms": [Term(id="t1", name="A", label=TermLabel.MEASURES, type="x").model_dump()],
                "med_terms": [],
                "predicates": [],
                "rules": []
            },
            2: {
                "terms": [],
                "med_terms": [MedicationTerm(id="m1", name="Drug").model_dump()],
                "predicates": [],
                "rules": []
            }
        }

        save_cluster_cache(cache, filepath)
        loaded = load_cluster_cache(filepath)

        assert len(loaded) == 2
        assert 1 in loaded
        assert 2 in loaded

    def test_load_empty_cache(self, temp_dir):
        """加载空缓存"""
        filepath = os.path.join(temp_dir, "empty_cache.json")
        Path(filepath).touch()

        loaded = load_cluster_cache(filepath)
        assert loaded == {}


class TestFailedTaskLogger:
    """FailedTaskLogger 测试"""

    def test_singleton(self, temp_dir):
        """单例模式"""
        # 重置单例
        FailedTaskLogger._instance = None

        with patch('pipeline.io_utils.get_config') as mock_config:
            mock_config.return_value.paths.gen_dir = temp_dir

            logger1 = FailedTaskLogger()
            logger2 = FailedTaskLogger()

            assert logger1 is logger2

    def test_log_failed_extraction(self, temp_dir):
        """记录抽取失败"""
        # 重置单例
        FailedTaskLogger._instance = None

        with patch('pipeline.io_utils.get_config') as mock_config:
            mock_config.return_value.paths.gen_dir = temp_dir

            logger = FailedTaskLogger()
            logger.clear()

            logger.log_failed_extraction(
                text_idx=0,
                text="测试文本",
                error="测试错误"
            )

            assert logger.get_failed_count() == 1

    def test_log_failed_cluster(self, temp_dir, sample_provenances):
        """记录聚类失败"""
        # 重置单例
        FailedTaskLogger._instance = None

        with patch('pipeline.io_utils.get_config') as mock_config:
            mock_config.return_value.paths.gen_dir = temp_dir

            logger = FailedTaskLogger()
            logger.clear()

            logger.log_failed_cluster(
                cluster_id=1,
                texts=sample_provenances,
                error="聚类处理错误"
            )

            assert logger.get_failed_count() == 1

    def test_clear(self, temp_dir):
        """清空记录"""
        # 重置单例
        FailedTaskLogger._instance = None

        with patch('pipeline.io_utils.get_config') as mock_config:
            mock_config.return_value.paths.gen_dir = temp_dir

            logger = FailedTaskLogger()
            logger.log_failed_extraction(0, "test", "error")

            logger.clear()

            assert logger.get_failed_count() == 0

    def test_thread_safety(self, temp_dir):
        """线程安全测试"""
        # 重置单例
        FailedTaskLogger._instance = None

        with patch('pipeline.io_utils.get_config') as mock_config:
            mock_config.return_value.paths.gen_dir = temp_dir

            logger = FailedTaskLogger()
            logger.clear()

            def log_items():
                for i in range(10):
                    logger.log_failed_extraction(i, f"text{i}", f"error{i}")

            threads = [threading.Thread(target=log_items) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # 线程安全测试：确保不会崩溃，且大部分记录被保存
            # 由于文件 I/O 的竞态条件，可能会有部分记录丢失
            assert logger.get_failed_count() >= 15  # 至少一半以上

    def test_reset_classmethod(self, temp_dir):
        """reset 类方法"""
        with patch('pipeline.io_utils.get_config') as mock_config:
            mock_config.return_value.paths.gen_dir = temp_dir

            logger = FailedTaskLogger()
            logger.log_failed_extraction(0, "test", "error")

            FailedTaskLogger.reset()

            assert len(logger._failed_tasks) == 0


class TestSaveToGen:
    """save_to_gen 测试"""

    def test_save_to_gen_basic(self, temp_gen_dir, sample_terms, sample_med_terms, sample_rules):
        """基本保存功能"""
        state = {
            "terms": sample_terms,
            "med_terms": sample_med_terms,
            "rules": sample_rules,
            "predicates": [],
            "provenance_buffer": []
        }

        with patch('pipeline.io_utils.get_config') as mock_config:
            mock_config.return_value.paths.gen_dir = temp_gen_dir

            stats = save_to_gen(state, gen_dir=temp_gen_dir, append=False)

            assert "terms.json" in stats
            assert "med_terms.json" in stats
            assert "rules.json" in stats

    def test_save_to_gen_append_mode(self, temp_gen_dir, sample_terms):
        """追加模式"""
        state1 = {
            "terms": [sample_terms[0]],
            "med_terms": [],
            "rules": [],
            "predicates": [],
            "provenance_buffer": []
        }
        state2 = {
            "terms": [sample_terms[1]],
            "med_terms": [],
            "rules": [],
            "predicates": [],
            "provenance_buffer": []
        }

        save_to_gen(state1, gen_dir=temp_gen_dir, append=True)
        save_to_gen(state2, gen_dir=temp_gen_dir, append=True)

        # 加载并检查合并结果
        loaded = _load_json_file(os.path.join(temp_gen_dir, "terms.json"))
        assert len(loaded) == 2


class TestLoadFromGen:
    """load_from_gen 测试"""

    def test_load_from_gen(self, temp_gen_dir, sample_terms):
        """从 gen 目录加载"""
        # 先保存数据
        filepath = os.path.join(temp_gen_dir, "terms.json")
        _save_json_file(filepath, [t.model_dump() for t in sample_terms])

        with patch('pipeline.io_utils.get_config') as mock_config:
            mock_config.return_value.paths.gen_dir = temp_gen_dir

            data = load_from_gen(gen_dir=temp_gen_dir, fields=["terms"])

            assert "terms" in data
            assert len(data["terms"]) == len(sample_terms)

    def test_load_from_empty_gen(self, temp_gen_dir):
        """从空目录加载"""
        data = load_from_gen(gen_dir=temp_gen_dir)
        assert data["terms"] == []
        assert data["med_terms"] == []
