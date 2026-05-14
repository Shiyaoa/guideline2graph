"""
术语标准化处理器 - 基于 OMOP CDM 的概念匹配 + LLM 审核

设计原则：
- Term 的 id 和 name 保持语义化，不被 OMOP 覆盖
- OMOP 匹配结果注册到独立的映射表 (TermMappingRegistry)
- 后处理阶段统一进行标准化和同义术语合并
"""
from typing import List, Dict, Optional, Any
import os
import re
import threading
import logging

from .models import (
    Term, TermLabel, MedicationTerm, Predicates,
    TermList, MedicationTermList, PredicatesList, TermExtractionResult
)
from .config import get_config, MatchConfig
from .term_mapping import get_registry

logger = logging.getLogger(__name__)


# 线程锁，保证 OMOP 单例在多线程环境下只初始化一次
_omop_lock = threading.Lock()
# LLM 审核器单例
_review_lock = threading.Lock()


def _get_omop_components():
    """延迟导入 OMOP normalizer 组件（线程安全单例模式）"""
    from omop_normalizer import OMOPMatcher, MappingCache

    # 双重检查锁定模式
    if not hasattr(_get_omop_components, "_matcher"):
        with _omop_lock:
            if not hasattr(_get_omop_components, "_matcher"):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                concept_csv_path = os.path.join(base_dir, "omop_normalizer", "CONCEPT.csv")
                cache_db_path = os.path.join(base_dir, "term_mapping_cache.db")

                logger.info(f"[OMOP] 加载 CONCEPT.csv: {concept_csv_path}")
                _get_omop_components._matcher = OMOPMatcher(concept_csv_path)
                _get_omop_components._cache = MappingCache(cache_db_path)

    return _get_omop_components._matcher, _get_omop_components._cache


def _preprocess_name_for_omop(name: str) -> str:
    """
    标准化抽取名称后再做 OMOP 匹配：去空白、常见中文→英文替换，提高命中率、减少误配。

    不改变上层 Term/MedicationTerm 对象，仅在匹配阶段使用。
    """
    if not name:
        return name
    s = " ".join(str(name).split())

    # 全小写精确命中中英词典（单成分药名等）
    from omop_normalizer.dictionaries import CHINESE_ENGLISH_DICT

    low = s.lower()
    if low in CHINESE_ENGLISH_DICT:
        return CHINESE_ENGLISH_DICT[low]

    # 子串替换：优先长词，避免「慢性肾」先于「慢性肾脏病」被截断
    if any("\u4e00" <= ch <= "\u9fff" for ch in s):
        keys = sorted(CHINESE_ENGLISH_DICT.keys(), key=len, reverse=True)
        for cn in keys:
            if cn in s:
                s = s.replace(cn, CHINESE_ENGLISH_DICT[cn])
        s = re.sub(r"\s+", " ", s).strip()
    return s


def _get_llm_reviewer():
    """延迟初始化 LLM 审核器（线程安全单例模式）"""
    from omop_normalizer.extractor import LLMTermExtractor

    if not hasattr(_get_llm_reviewer, "_extractor"):
        with _review_lock:
            if not hasattr(_get_llm_reviewer, "_extractor"):
                cfg = get_config().llm
                _get_llm_reviewer._extractor = LLMTermExtractor(
                    api_type="qwen",  # 默认使用阿里云
                    api_key=cfg.api_key,
                    model=cfg.model,
                    base_url=cfg.base_url
                )

    return _get_llm_reviewer._extractor


class MatchResult:
    """OMOP 概念匹配结果"""
    def __init__(
        self, matched: bool, concept_id: int = 0, concept_name: str = "",
        domain_id: str = "", vocabulary_id: str = "", match_score: float = 0.0,
        match_type: str = ""
    ):
        self.matched = matched
        self.concept_id = concept_id
        self.concept_name = concept_name
        self.domain_id = domain_id
        self.vocabulary_id = vocabulary_id
        self.match_score = match_score
        self.match_type = match_type


class TermProcessor:
    """术语处理器 - 映射到 OMOP CDM 标准概念"""

    def __init__(self, config: Optional[MatchConfig] = None, enable_cache: bool = True):
        self._config = config or get_config().match
        self._enable_cache = enable_cache
        self._matcher, self._cache = _get_omop_components()

    def process(self, response: TermList) -> TermList:
        """处理术语列表，通过 OMOP 概念匹配进行标准化"""
        processed = []
        for term in response.items:
            entity_type = term.label.value if isinstance(term.label, TermLabel) else str(term.label)
            match_result = self._match_term(term.name, entity_type)
            if match_result.matched:
                processed.append(self._create_term(match_result, term))
            else:
                processed.append(term)
        return TermList(items=processed)

    def _match_term(self, term_name: str, entity_type: str) -> MatchResult:
        """匹配单个术语到 OMOP 概念"""
        term_name = _preprocess_name_for_omop(term_name)
        if self._enable_cache:
            cached = self._cache.get(term_name, entity_type)
            if cached and cached.get("concept_id"):
                return MatchResult(
                    matched=True, concept_id=cached["concept_id"],
                    concept_name=cached["concept_name"], domain_id=cached["domain_id"],
                    vocabulary_id=cached["vocabulary_id"],
                    match_score=cached.get("match_score", 100), match_type="cached"
                )

        domain_map = {
            "measures": "Measurement", "conditions": "Condition",
            "procedures": "Procedure", "observations": "Observation"
        }
        domain_id = domain_map.get(entity_type.lower(), None)
        matches = self._matcher.match(term_name, domain_id=domain_id, top_k=1)

        if matches and matches[0].match_score >= self._config.term_threshold:
            best = matches[0]
            result = MatchResult(
                matched=True, concept_id=best.concept_id,
                concept_name=best.concept_name, domain_id=best.domain_id,
                vocabulary_id=best.vocabulary_id, match_score=best.match_score,
                match_type=best.match_type
            )
            if self._enable_cache and result.match_type != "cached":
                self._cache.save({
                    "chinese_term": term_name, "english_term": term_name,
                    "concept_id": result.concept_id, "concept_name": result.concept_name,
                    "domain_id": result.domain_id, "vocabulary_id": result.vocabulary_id,
                    "match_score": result.match_score, "match_type": result.match_type
                }, source="pipeline")
            return result
        return MatchResult(matched=False)

    @staticmethod
    def _create_term(match: MatchResult, original: Term) -> Term:
        """从 OMOP 概念创建 Term 对象"""
        code_bindings = list(original.code_bindings or [])
        code_bindings.append({
            "type": "OMOPConcept",
            "code": match.concept_id,
            "display": match.concept_name,
            "system": match.vocabulary_id,
            "confidence": min(max(match.match_score / 100.0, 0.0), 1.0),
        })
        return Term(
            id=str(match.concept_id),
            name=match.concept_name,
            label=original.label,
            type=original.type or match.vocabulary_id,
            clinical_entity=original.clinical_entity or original.id,
            concept=match.concept_name,
            value_domain=original.value_domain,
            unit=original.unit,
            value_set_binding=original.value_set_binding,
            code_bindings=code_bindings,
            data_bindings=original.data_bindings,
            fhir_binding_hint=original.fhir_binding_hint,
            omop_binding_hint={
                **(original.omop_binding_hint or {}),
                "concept_id": match.concept_id,
                "concept_name": match.concept_name,
                "domain_id": match.domain_id,
                "vocabulary_id": match.vocabulary_id,
            },
            normalization=original.normalization,
            normalization_confidence=min(max(match.match_score / 100.0, 0.0), 1.0),
            source_evidence=original.source_evidence,
        )


class MedicationProcessor:
    """药物处理器 - 映射到 OMOP CDM 标准概念（Drug 领域）"""

    def __init__(self, config: Optional[MatchConfig] = None, enable_cache: bool = True):
        self._config = config or get_config().match
        self._enable_cache = enable_cache
        self._matcher, self._cache = _get_omop_components()

    def process(self, response: MedicationTermList) -> MedicationTermList:
        """处理药物术语列表，通过 OMOP 概念匹配进行标准化"""
        processed = []
        for med in response.items:
            match_result = self._match_med(med.name)
            if match_result.matched:
                processed.append(self._create_med(match_result, med))
            else:
                processed.append(med)
        return MedicationTermList(items=processed)

    def _match_med(self, med_name: str) -> MatchResult:
        """匹配单个药物术语到 OMOP 概念"""
        med_name = _preprocess_name_for_omop(med_name)
        if self._enable_cache:
            cached = self._cache.get(med_name, "Drug")
            if cached and cached.get("concept_id"):
                return MatchResult(
                    matched=True, concept_id=cached["concept_id"],
                    concept_name=cached["concept_name"], domain_id=cached["domain_id"],
                    vocabulary_id=cached["vocabulary_id"],
                    match_score=cached.get("match_score", 100), match_type="cached"
                )

        matches = self._matcher.match(med_name, domain_id="Drug", top_k=1)

        if matches and matches[0].match_score >= self._config.med_threshold:
            best = matches[0]
            result = MatchResult(
                matched=True, concept_id=best.concept_id,
                concept_name=best.concept_name, domain_id=best.domain_id,
                vocabulary_id=best.vocabulary_id, match_score=best.match_score,
                match_type=best.match_type
            )
            if self._enable_cache and result.match_type != "cached":
                self._cache.save({
                    "chinese_term": med_name, "english_term": med_name,
                    "concept_id": result.concept_id, "concept_name": result.concept_name,
                    "domain_id": result.domain_id, "vocabulary_id": result.vocabulary_id,
                    "match_score": result.match_score, "match_type": result.match_type
                }, source="pipeline")
            return result
        return MatchResult(matched=False)

    @staticmethod
    def _create_med(match: MatchResult, original: MedicationTerm) -> MedicationTerm:
        """从 OMOP 概念创建 MedicationTerm 对象"""
        code_bindings = list(original.code_bindings or [])
        code_bindings.append({
            "type": "OMOPConcept",
            "code": match.concept_id,
            "display": match.concept_name,
            "system": match.vocabulary_id,
            "confidence": min(max(match.match_score / 100.0, 0.0), 1.0),
        })
        return MedicationTerm(
            id=str(match.concept_id),
            name=match.concept_name,
            drug_class=original.drug_class,
            subclass=original.subclass,
            clinical_entity=original.clinical_entity or original.id,
            concept=match.concept_name,
            value_set_binding=original.value_set_binding,
            code_bindings=code_bindings,
            data_bindings=original.data_bindings,
            fhir_binding_hint=original.fhir_binding_hint,
            omop_binding_hint={
                **(original.omop_binding_hint or {}),
                "concept_id": match.concept_id,
                "concept_name": match.concept_name,
                "domain_id": match.domain_id,
                "vocabulary_id": match.vocabulary_id,
            },
            normalization_confidence=min(max(match.match_score / 100.0, 0.0), 1.0),
            source_evidence=original.source_evidence,
        )


class CombinedTermProcessor:
    """合并术语处理器 - 同时处理术语和药物，基于 OMOP CDM 匹配 + LLM 审核"""

    def __init__(self, config: Optional[MatchConfig] = None, enable_cache: bool = True,
                 enable_review: bool = True):
        self._config = config or get_config().match
        self._enable_cache = enable_cache
        self._enable_review = enable_review and self._config.enable_review
        self._matcher, self._cache = _get_omop_components()

    def process(self, result: TermExtractionResult) -> TermExtractionResult:
        """处理合并的抽取结果，注册 OMOP 映射但不修改 Term"""
        registry = get_registry()
        processed_terms = []
        processed_meds = []

        for term in result.terms:
            match_result = self._match_in_omop(
                term.name, domain_hints=["Measurement", "Condition", "Procedure", "Observation"]
            )
            if match_result.matched:
                # 注册映射到映射表
                registry.register_term(
                    semantic_id=term.id,
                    concept_id=match_result.concept_id,
                    concept_name=match_result.concept_name,
                    domain_id=match_result.domain_id,
                    vocabulary_id=match_result.vocabulary_id,
                    match_score=match_result.match_score,
                    match_type=match_result.match_type
                )
            # 保持 Term 原样不变
            processed_terms.append(term)

        for med in result.med_terms:
            match_result = self._match_in_omop(med.name, domain_hints=["Drug"])
            if match_result.matched:
                # 注册映射到映射表
                registry.register_med(
                    semantic_id=med.id,
                    concept_id=match_result.concept_id,
                    concept_name=match_result.concept_name,
                    domain_id=match_result.domain_id,
                    vocabulary_id=match_result.vocabulary_id,
                    match_score=match_result.match_score,
                    match_type=match_result.match_type
                )
            # 保持 MedicationTerm 原样不变
            processed_meds.append(med)

        return TermExtractionResult(terms=processed_terms, med_terms=processed_meds)

    def _match_in_omop(self, name: str, domain_hints: Optional[List[str]] = None,
                        original_term: str = "") -> MatchResult:
        """在 OMOP 中搜索匹配，可选 LLM 审核"""
        name = _preprocess_name_for_omop(name)

        # 缓存检查（缓存的数据已经是审核通过的）
        if self._enable_cache:
            for domain in (domain_hints or ["Measurement", "Condition", "Procedure", "Observation", "Drug"]):
                cached = self._cache.get(name, domain)
                if cached and cached.get("concept_id"):
                    return MatchResult(
                        matched=True, concept_id=cached["concept_id"],
                        concept_name=cached["concept_name"], domain_id=cached["domain_id"],
                        vocabulary_id=cached["vocabulary_id"],
                        match_score=cached.get("match_score", 100), match_type="cached"
                    )

        # OMOP 匹配（Drug 域使用 med_threshold，其余使用 term_threshold）
        for domain_id in (domain_hints or ["Measurement", "Condition", "Procedure", "Observation", "Drug"]):
            min_score = (
                self._config.med_threshold if domain_id == "Drug" else self._config.term_threshold
            )
            matches = self._matcher.match(name, domain_id=domain_id, top_k=1)
            if matches and matches[0].match_score >= min_score:
                best = matches[0]

                # LLM 审核检查
                if self._enable_review and best.match_score < self._config.high_confidence_skip_review:
                    review_result = self._review_match(name, best.concept_name, best.domain_id)
                    if not review_result.is_correct:
                        # 审核不通过，返回未匹配（保留原始术语）
                        logger.info(
                            f"[OMOP审核] 拒绝匹配: '{name}' -> '{best.concept_name}' "
                            f"(分数:{best.match_score:.1f}, 原因:{review_result.reason[:50]})"
                        )
                        continue  # 尝试下一个 domain
                    logger.info(f"[OMOP审核] 通过: '{name}' -> '{best.concept_name}' (分数:{best.match_score:.1f})")

                # 匹配成功，存入缓存
                result = MatchResult(
                    matched=True, concept_id=best.concept_id,
                    concept_name=best.concept_name, domain_id=best.domain_id,
                    vocabulary_id=best.vocabulary_id, match_score=best.match_score,
                    match_type=best.match_type
                )
                if self._enable_cache:
                    self._cache.save({
                        "chinese_term": name, "english_term": name,
                        "concept_id": result.concept_id, "concept_name": result.concept_name,
                        "domain_id": result.domain_id, "vocabulary_id": result.vocabulary_id,
                        "match_score": result.match_score, "match_type": result.match_type
                    }, source="pipeline")
                return result
        return MatchResult(matched=False)

    def _review_match(self, term: str, concept_name: str, domain: str):
        """调用 LLM 审核匹配结果"""
        try:
            reviewer = _get_llm_reviewer()
            return reviewer.review_match(
                chinese_term=term,
                english_term=term,
                concept_name=concept_name,
                domain=domain
            )
        except Exception as e:
            logger.warning(f"[OMOP审核] 审核失败: {e}")
            # 审核失败时保守处理：认为匹配有效
            from omop_normalizer.models import ReviewResult
            return ReviewResult(is_correct=True, reason=f"审核异常: {e}", confidence=0.5)


class PredicateProcessor:
    """谓词处理器"""

    def __init__(self, config: Optional[MatchConfig] = None):
        self._config = config or get_config().match

    def process(self, response: PredicatesList) -> PredicatesList:
        """处理谓词列表（当前直接返回原列表）"""
        return response


# 便捷函数
def process_terms(response: TermList) -> TermList:
    """处理术语列表 - 映射到 OMOP 概念"""
    return TermProcessor().process(response)


def process_med_terms(response: MedicationTermList) -> MedicationTermList:
    """处理药物术语列表 - 映射到 OMOP 概念"""
    return MedicationProcessor().process(response)


def process_combined_terms(result: TermExtractionResult) -> TermExtractionResult:
    """处理合并的术语抽取结果"""
    return CombinedTermProcessor().process(result)


def process_predicates(response: PredicatesList) -> PredicatesList:
    """处理谓词列表"""
    return PredicateProcessor().process(response)
