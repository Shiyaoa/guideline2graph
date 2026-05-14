"""
中文医学术语标准化器

主入口，整合LLM提取、概念匹配、缓存管理
"""

import time
from typing import List, Optional
from dataclasses import asdict

from .matcher import OMOPMatcher
from .extractor import LLMTermExtractor
from .cache import MappingCache
from .models import ExtractedTerm, MappingResult, ConceptMatch


class ChineseTermNormalizer:
    """中文医学术语标准化器"""

    def __init__(self,
                 concept_csv_path: str,
                 cache_db_path: str = "term_mapping_cache.db",
                 llm_api_type: str = "qwen",
                 llm_api_key: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 confidence_threshold: float = 0.75,
                 enable_review: bool = True,
                 verbose: bool = True):
        """
        初始化标准化器

        Args:
            concept_csv_path: OMOP CONCEPT.csv文件路径
            cache_db_path: 映射缓存数据库路径
            llm_api_type: LLM API类型 (qwen, deepseek, openai, anthropic)
            llm_api_key: LLM API密钥
            llm_model: LLM模型名称
            confidence_threshold: 置信度阈值，低于此值标记为需审核
            enable_review: 是否启用LLM审核
            verbose: 是否输出详细日志
        """
        # 初始化组件
        self.matcher = OMOPMatcher(concept_csv_path)
        self.extractor = LLMTermExtractor(
            api_type=llm_api_type,
            api_key=llm_api_key,
            model=llm_model
        )
        self.cache = MappingCache(cache_db_path)

        # 配置
        self.confidence_threshold = confidence_threshold
        self.enable_review = enable_review
        self.verbose = verbose

    def _log(self, message: str):
        """输出日志"""
        if self.verbose:
            print(message)

    def normalize(self, text: str, source: str = "") -> List[MappingResult]:
        """
        标准化文本中的医学术语

        Args:
            text: 中文临床指南文本
            source: 来源标识（用于追溯）

        Returns:
            标准化结果列表
        """
        results = []

        # Step 1: LLM提取术语
        self._log("\n[Step 1] LLM提取术语...")
        terms = self.extractor.extract_terms(text)

        if not terms:
            self._log("  未提取到术语")
            return results

        self._log(f"  提取到 {len(terms)} 个术语")

        for term in terms:
            result = self._process_term(term, source)
            if result:
                results.append(result)

        return results

    def _process_term(self, term: ExtractedTerm, source: str) -> Optional[MappingResult]:
        """处理单个术语"""
        self._log(f"\n处理: {term.chinese_text} ({term.entity_type})")

        # Step 2: 检查缓存
        cached = self.cache.get(term.chinese_text, term.entity_type)
        if cached and cached.get('concept_id'):
            self._log(f"  [缓存命中] -> {cached['concept_id']}")
            review_confidence = cached.get('review_confidence', 1.0)
            final_confidence = cached.get('final_confidence')
            if final_confidence is None:
                final_confidence = (
                    term.confidence +
                    cached.get('match_score', 100) / 100 +
                    review_confidence
                ) / 3
            return MappingResult(
                chinese_term=term.chinese_text,
                english_term=cached['english_term'],
                concept_id=cached['concept_id'],
                concept_name=cached['concept_name'],
                domain_id=cached['domain_id'],
                vocabulary_id=cached['vocabulary_id'],
                extraction_confidence=term.confidence,
                match_score=cached.get('match_score', 100),
                review_confidence=review_confidence,
                final_confidence=final_confidence,
                status='cached',
                match_type='cached'
            )

        # Step 3: 匹配OMOP概念
        self._log(f"  [Step 2] 匹配OMOP概念...")
        matches = self.matcher.match(term.english_standard, term.entity_type)

        if not matches:
            self._log(f"  未找到匹配")
            return MappingResult(
                chinese_term=term.chinese_text,
                english_term=term.english_standard,
                concept_id=0,
                concept_name="",
                domain_id=term.entity_type,
                vocabulary_id="",
                extraction_confidence=term.confidence,
                match_score=0,
                review_confidence=0,
                final_confidence=0,
                status='no_match',
                match_type='none'
            )

        best = matches[0]
        self._log(f"  最佳匹配: {best.concept_name[:50]} (ID:{best.concept_id}, 分数:{best.match_score:.1f})")

        # Step 4: LLM审核
        review_confidence = 1.0
        is_correct = True

        if self.enable_review:
            self._log(f"  [Step 3] LLM审核...")
            review = self.extractor.review_match(
                term.chinese_text,
                term.english_standard,
                best.concept_name,
                best.domain_id,
                term.source_text
            )
            review_confidence = review.confidence
            is_correct = review.is_correct

            if is_correct:
                self._log(f"  [OK] 审核通过 - {review.reason[:50]}")
            else:
                self._log(f"  [FAIL] 审核不通过 - {review.reason[:50]}")

        # 计算最终置信度
        final_confidence = (term.confidence + best.match_score / 100 + review_confidence) / 3

        # 确定状态
        if not is_correct:
            status = 'rejected'
        elif final_confidence >= self.confidence_threshold:
            status = 'approved'
        else:
            status = 'needs_review'

        # 创建结果
        result = MappingResult(
            chinese_term=term.chinese_text,
            english_term=term.english_standard,
            concept_id=best.concept_id,
            concept_name=best.concept_name,
            domain_id=best.domain_id,
            vocabulary_id=best.vocabulary_id,
            extraction_confidence=term.confidence,
            match_score=best.match_score,
            review_confidence=review_confidence,
            final_confidence=final_confidence,
            status=status,
            match_type=best.match_type
        )

        # 审核通过则存入缓存
        if status == 'approved':
            self.cache.save(result.to_dict(), source)
            self._log(f"  已存入缓存")

        return result

    def batch_normalize(self, texts: List[tuple]) -> List[MappingResult]:
        """
        批量标准化

        Args:
            texts: [(text, source), ...] 元组列表

        Returns:
            所有标准化结果列表
        """
        all_results = []

        for i, (text, source) in enumerate(texts):
            self._log(f"\n{'=' * 60}")
            self._log(f"处理文档 {i + 1}/{len(texts)}: {source[:30]}...")
            self._log(f"{'=' * 60}")

            results = self.normalize(text, source)
            all_results.extend(results)

            # 显示进度统计
            stats = self.cache.get_statistics()
            self._log(f"\n当前缓存: {stats['total_mappings']} 条映射")

            # 避免API限流
            time.sleep(0.3)

        return all_results

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return self.cache.get_statistics()

    def export_results(self, results: List[MappingResult], output_path: str):
        """
        导出结果到Excel

        Args:
            results: 结果列表
            output_path: 输出文件路径
        """
        import pandas as pd
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_excel(output_path, index=False)
        self._log(f"结果已保存到 {output_path}")

    def search_concept(self, keyword: str, domain_id: Optional[str] = None,
                       top_k: int = 10) -> List[ConceptMatch]:
        """
        搜索概念（工具方法）

        Args:
            keyword: 搜索关键词
            domain_id: 可选的领域限制
            top_k: 返回数量

        Returns:
            匹配的概念列表
        """
        return self.matcher.match(keyword, domain_id, top_k)


# ==================== 便捷函数 ====================

def create_normalizer(concept_csv_path: str = "CONCEPT.csv",
                      api_type: str = "qwen",
                      **kwargs) -> ChineseTermNormalizer:
    """
    创建标准化器的便捷函数

    Args:
        concept_csv_path: OMOP CONCEPT.csv路径
        api_type: LLM API类型
        **kwargs: 其他参数传递给ChineseTermNormalizer

    Returns:
        初始化好的标准化器
    """
    return ChineseTermNormalizer(
        concept_csv_path=concept_csv_path,
        llm_api_type=api_type,
        **kwargs
    )