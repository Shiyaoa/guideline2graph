"""
术语-OMOP 映射注册表

维护语义化术语 ID 与 OMOP 概念的映射关系，
支持同义术语识别和后处理标准化。
"""
import json
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class TermOMOPMapping:
    """术语到 OMOP 概念的映射"""
    semantic_id: str           # 语义化 ID（如 "meas.egfr"）
    concept_id: int            # OMOP 概念 ID
    concept_name: str          # OMOP 完整名称
    domain_id: str             # 领域
    vocabulary_id: str         # 词汇表
    match_score: float         # 匹配分数
    match_type: str            # 匹配类型: exact/fuzzy/special/cached


class TermMappingRegistry:
    """
    术语-OMOP 映射注册表（线程安全单例）

    职责：
    1. 注册术语到 OMOP 概念的映射
    2. 提供反向索引：concept_id -> [semantic_ids]
    3. 识别同义术语（映射到同一 OMOP 概念）
    4. 持久化映射表到 JSON
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 核心映射：semantic_id -> TermOMOPMapping
        self._mappings: Dict[str, TermOMOPMapping] = {}

        # 反向索引：concept_id -> [semantic_ids]
        self._concept_to_terms: Dict[int, List[str]] = {}

        # 药物映射
        self._med_mappings: Dict[str, TermOMOPMapping] = {}
        self._med_concept_to_terms: Dict[int, List[str]] = {}

        self._initialized = True

    def register_term(self, semantic_id: str, concept_id: int,
                      concept_name: str, domain_id: str,
                      vocabulary_id: str, match_score: float,
                      match_type: str) -> None:
        """
        注册非药物术语映射

        Args:
            semantic_id: 语义化 ID（如 "meas.egfr"）
            concept_id: OMOP 概念 ID
            concept_name: OMOP 完整名称
            domain_id: OMOP 领域
            vocabulary_id: OMOP 词汇表
            match_score: 匹配分数
            match_type: 匹配类型
        """
        mapping = TermOMOPMapping(
            semantic_id=semantic_id,
            concept_id=concept_id,
            concept_name=concept_name,
            domain_id=domain_id,
            vocabulary_id=vocabulary_id,
            match_score=match_score,
            match_type=match_type
        )
        self._mappings[semantic_id] = mapping

        # 更新反向索引
        if concept_id not in self._concept_to_terms:
            self._concept_to_terms[concept_id] = []
        if semantic_id not in self._concept_to_terms[concept_id]:
            self._concept_to_terms[concept_id].append(semantic_id)

    def register_med(self, semantic_id: str, concept_id: int,
                     concept_name: str, domain_id: str,
                     vocabulary_id: str, match_score: float,
                     match_type: str) -> None:
        """注册药物术语映射"""
        mapping = TermOMOPMapping(
            semantic_id=semantic_id,
            concept_id=concept_id,
            concept_name=concept_name,
            domain_id=domain_id,
            vocabulary_id=vocabulary_id,
            match_score=match_score,
            match_type=match_type
        )
        self._med_mappings[semantic_id] = mapping

        if concept_id not in self._med_concept_to_terms:
            self._med_concept_to_terms[concept_id] = []
        if semantic_id not in self._med_concept_to_terms[concept_id]:
            self._med_concept_to_terms[concept_id].append(semantic_id)

    def get_term_mapping(self, semantic_id: str) -> Optional[TermOMOPMapping]:
        """获取术语映射"""
        return self._mappings.get(semantic_id)

    def get_med_mapping(self, semantic_id: str) -> Optional[TermOMOPMapping]:
        """获取药物映射"""
        return self._med_mappings.get(semantic_id)

    def get_term_synonyms(self, semantic_id: str) -> List[str]:
        """
        获取映射到同一 OMOP 概念的同义术语

        Args:
            semantic_id: 语义化 ID

        Returns:
            同义术语 ID 列表（不包含自身）
        """
        mapping = self._mappings.get(semantic_id)
        if not mapping:
            return []

        concept_id = mapping.concept_id
        synonyms = self._concept_to_terms.get(concept_id, [])
        return [s for s in synonyms if s != semantic_id]

    def get_med_synonyms(self, semantic_id: str) -> List[str]:
        """获取映射到同一 OMOP 概念的同义药物"""
        mapping = self._med_mappings.get(semantic_id)
        if not mapping:
            return []

        concept_id = mapping.concept_id
        synonyms = self._med_concept_to_terms.get(concept_id, [])
        return [s for s in synonyms if s != semantic_id]

    def get_all_term_groups(self) -> Dict[int, List[str]]:
        """
        获取所有术语分组（按 concept_id）

        Returns:
            {concept_id: [semantic_ids]} 映射
        """
        # 只返回有多个同义术语的分组
        return {
            cid: sids for cid, sids in self._concept_to_terms.items()
            if len(sids) > 1
        }

    def get_all_med_groups(self) -> Dict[int, List[str]]:
        """获取所有药物分组（按 concept_id）"""
        return {
            cid: sids for cid, sids in self._med_concept_to_terms.items()
            if len(sids) > 1
        }

    def get_primary_term_id(self, semantic_id: str) -> str:
        """
        获取同义术语组中的主 ID（match_score 最高的）

        Args:
            semantic_id: 语义化 ID

        Returns:
            主 ID（如果存在同义术语）
        """
        synonyms = self.get_term_synonyms(semantic_id)
        if not synonyms:
            return semantic_id

        # 包含自身，选择 match_score 最高的
        all_ids = [semantic_id] + synonyms
        best_id = max(all_ids, key=lambda sid: self._mappings[sid].match_score)
        return best_id

    def get_primary_med_id(self, semantic_id: str) -> str:
        """获取同义药物组中的主 ID"""
        synonyms = self.get_med_synonyms(semantic_id)
        if not synonyms:
            return semantic_id

        all_ids = [semantic_id] + synonyms
        best_id = max(all_ids, key=lambda sid: self._med_mappings[sid].match_score)
        return best_id

    def save(self, path: str) -> None:
        """
        保存映射表到 JSON

        Args:
            path: 输出文件路径
        """
        data = {
            "terms": {
                sid: asdict(mapping)
                for sid, mapping in self._mappings.items()
            },
            "med_terms": {
                sid: asdict(mapping)
                for sid, mapping in self._med_mappings.items()
            },
            "term_groups": {
                str(cid): sids
                for cid, sids in self._concept_to_terms.items()
                if len(sids) > 1
            },
            "med_groups": {
                str(cid): sids
                for cid, sids in self._med_concept_to_terms.items()
                if len(sids) > 1
            }
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """
        从 JSON 加载映射表

        Args:
            path: 输入文件路径
        """
        if not Path(path).exists():
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 加载术语映射
        for sid, mapping_dict in data.get("terms", {}).items():
            mapping = TermOMOPMapping(**mapping_dict)
            self._mappings[sid] = mapping

            if mapping.concept_id not in self._concept_to_terms:
                self._concept_to_terms[mapping.concept_id] = []
            if sid not in self._concept_to_terms[mapping.concept_id]:
                self._concept_to_terms[mapping.concept_id].append(sid)

        # 加载药物映射
        for sid, mapping_dict in data.get("med_terms", {}).items():
            mapping = TermOMOPMapping(**mapping_dict)
            self._med_mappings[sid] = mapping

            if mapping.concept_id not in self._med_concept_to_terms:
                self._med_concept_to_terms[mapping.concept_id] = []
            if sid not in self._med_concept_to_terms[mapping.concept_id]:
                self._med_concept_to_terms[mapping.concept_id].append(sid)

    def clear(self) -> None:
        """清空映射表（用于测试）"""
        self._mappings.clear()
        self._concept_to_terms.clear()
        self._med_mappings.clear()
        self._med_concept_to_terms.clear()

    @classmethod
    def reset_instance(cls):
        """重置单例实例（用于测试）"""
        with cls._lock:
            cls._instance = None

    # 统计信息
    @property
    def term_count(self) -> int:
        return len(self._mappings)

    @property
    def med_count(self) -> int:
        return len(self._med_mappings)

    @property
    def synonym_group_count(self) -> int:
        return len(self.get_all_term_groups()) + len(self.get_all_med_groups())


# 便捷函数
def get_registry() -> TermMappingRegistry:
    """获取映射注册表单例"""
    return TermMappingRegistry()