"""
数据模型定义
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class ExtractedTerm:
    """LLM提取的术语"""
    chinese_text: str           # 中文术语
    entity_type: str            # 实体类型 (Condition/Drug/Procedure/Measurement/Observation)
    english_standard: str       # LLM给出的标准英文名
    confidence: float           # 提取置信度
    source_text: str            # 来源文本上下文


@dataclass
class ConceptMatch:
    """OMOP概念匹配结果"""
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    match_score: float
    match_type: str  # exact, synonym, abbreviation, drug_class, fuzzy, special
    is_standard: bool


@dataclass
class MappingResult:
    """最终映射结果"""
    chinese_term: str
    english_term: str
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    extraction_confidence: float
    match_score: float
    review_confidence: float
    final_confidence: float
    status: str  # approved, rejected, needs_review, no_match, cached
    match_type: str

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "chinese_term": self.chinese_term,
            "english_term": self.english_term,
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "domain_id": self.domain_id,
            "vocabulary_id": self.vocabulary_id,
            "extraction_confidence": self.extraction_confidence,
            "match_score": self.match_score,
            "review_confidence": self.review_confidence,
            "final_confidence": self.final_confidence,
            "status": self.status,
            "match_type": self.match_type,
        }


@dataclass
class ReviewResult:
    """LLM审核结果"""
    is_correct: bool
    correct_concept_name: str
    reason: str
    confidence: float


@dataclass
class TermMapping:
    """映射缓存记录"""
    chinese_term: str
    english_term: str
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    confidence: float
    match_type: str
    is_verified: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None