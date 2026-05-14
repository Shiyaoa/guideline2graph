"""
OMOP概念匹配器

支持多种匹配策略：
- 特殊概念映射
- 缩写扩展
- 中文词典映射
- 精确匹配
- 包含匹配
- 模糊匹配 (rapidfuzz)
- 多关键词组合
- 药物类别智能匹配（利用 CONCEPT_ANCESTOR 层级关系）

优化：
- 支持 pickle 缓存，加载速度从 120s 降至 ~3s
- 自动区分药物类别（ATC）和具体药物（Ingredient）
- 模糊匹配默认偏速度：rapidfuzz 限条数 + 高分跳过词序列 + 仅前 M 条做词级融合（见 CLAUDE.md）
"""

import pandas as pd
import pickle
import os
import re
from typing import List, Optional, Set, Dict
from dataclasses import dataclass

from .models import ConceptMatch
from .dictionaries import (
    ABBREVIATION_DICT,
    CHINESE_ENGLISH_DICT,
    SPECIAL_MAPPINGS,
)
from .token_sequence_match import combined_fuzzy_score

# 词级对齐 + WRatio 融合后的最低接受线（与旧版「WRatio≥70」并用，见 match()）
_FUZZY_COMBINED_MIN = 67.0

# 模糊匹配剪枝（默认偏速度；与 CLAUDE.md「OMOP 模糊匹配剪枝」一致）
_FUZZY_EXTRACT_LIMIT = 22
_FUZZY_WRATIO_FLOOR = 58.0
_FUZZY_SKIP_TOKEN_ALIGN_IF_WRATIO_GE = 92.0
_FUZZY_TOKEN_ALIGN_TOP_M = 12
_FUZZY_TAIL_WRATIO_MIN = 70.0


class OMOPMatcher:
    """OMOP概念匹配器"""

    # 只保留这些领域（临床相关）
    RELEVANT_DOMAINS = {'Drug', 'Measurement', 'Condition', 'Procedure', 'Observation', 'Device'}

    # 药物类别关键词（用于判断术语是否为类别）
    DRUG_CLASS_KEYWORDS = {
        'inhibitors', 'inhibitor', 'antagonists', 'antagonist',
        'agonists', 'agonist', 'blockers', 'blocker',
        'diuretics', 'diuretic', 'analogs', 'analogue', 'analogues',
        'hormones', 'hormone', 'agents', 'agent',
        'sulfonylureas', 'thiazolidinediones', 'biguanides',
        'statins', 'anticoagulants', 'nsaids',
    }

    def __init__(self, concept_csv_path: str, use_cache: bool = True,
                 load_ancestor: bool = True):
        """
        初始化匹配器

        Args:
            concept_csv_path: OMOP CONCEPT.csv文件路径
            use_cache: 是否使用 pickle 缓存（默认 True）
            load_ancestor: 是否加载 CONCEPT_ANCESTOR 表（用于判断层级关系）
        """
        self._concept_csv_path = concept_csv_path
        self._load_ancestor = load_ancestor

        # 确定缓存路径
        cache_path = concept_csv_path.replace('.csv', '_optimized.pkl')
        ancestor_cache_path = concept_csv_path.replace('CONCEPT.csv', 'CONCEPT_ANCESTOR_optimized.pkl')

        # 尝试从缓存加载
        if use_cache and os.path.exists(cache_path):
            print(f"[OMOP] 从缓存加载: {cache_path}")
            self._load_from_cache(cache_path)
        else:
            self._load_concept_table(concept_csv_path, use_cache, cache_path)

        # 加载 CONCEPT_ANCESTOR 表
        self.ancestor_df = None
        self._descendant_cache: Dict[int, int] = {}  # concept_id -> 后代数量
        if load_ancestor:
            ancestor_csv_path = concept_csv_path.replace('CONCEPT.csv', 'CONCEPT_ANCESTOR.csv')
            if os.path.exists(ancestor_csv_path):
                self._load_ancestor_table(ancestor_csv_path, use_cache, ancestor_cache_path)

    def _load_concept_table(self, concept_csv_path: str, use_cache: bool, cache_path: str):
        """加载 CONCEPT 表"""
        print("加载OMOP CONCEPT表...")
        # 读取需要的列，包括 concept_class_id
        usecols = ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
                   'standard_concept', 'invalid_reason', 'concept_class_id']
        self.all_concepts = pd.read_csv(
            concept_csv_path, engine='python', on_bad_lines='skip', usecols=usecols
        )

        # 标准概念 (standard_concept='S') + 只保留相关领域
        self.standard = self.all_concepts[
            (self.all_concepts['standard_concept'] == 'S') &
            (self.all_concepts['invalid_reason'].isna()) &
            (self.all_concepts['domain_id'].isin(self.RELEVANT_DOMAINS))
        ].copy()

        # 分类概念 (standard_concept='C') - 如ATC药物类别
        self.classification = self.all_concepts[
            (self.all_concepts['standard_concept'] == 'C') &
            (self.all_concepts['invalid_reason'].isna()) &
            (self.all_concepts['domain_id'].isin(self.RELEVANT_DOMAINS))
        ].copy()

        # 所有有效概念（标准+分类）
        self.valid = pd.concat([self.standard, self.classification]).drop_duplicates()

        # 释放内存
        del self.all_concepts

        print(f"标准概念: {len(self.standard):,} 个")
        print(f"分类概念: {len(self.classification):,} 个")
        print(f"有效概念总数: {len(self.valid):,} 个")

        # 构建索引
        self._build_indexes()

        # 保存到缓存
        if use_cache:
            self._save_to_cache(cache_path)

    def _load_ancestor_table(self, ancestor_csv_path: str, use_cache: bool, cache_path: str):
        """加载 CONCEPT_ANCESTOR 表"""
        if use_cache and os.path.exists(cache_path):
            print(f"[OMOP] 从缓存加载 CONCEPT_ANCESTOR: {cache_path}")
            import time
            start = time.time()
            with open(cache_path, 'rb') as f:
                self.ancestor_df = pickle.load(f)
            print(f"[OMOP] CONCEPT_ANCESTOR 加载完成: {time.time() - start:.2f}s")
        else:
            print("加载 CONCEPT_ANCESTOR 表...")
            import time
            start = time.time()
            usecols = ['ancestor_concept_id', 'descendant_concept_id']
            self.ancestor_df = pd.read_csv(ancestor_csv_path, usecols=usecols)
            print(f"[OMOP] CONCEPT_ANCESTOR 加载完成: {time.time() - start:.2f}s, {len(self.ancestor_df):,} 条记录")

            # 构建后代数量缓存
            print("[OMOP] 构建后代数量索引...")
            descendant_counts = self.ancestor_df.groupby('ancestor_concept_id').size()
            for concept_id, count in descendant_counts.items():
                self._descendant_cache[int(concept_id)] = int(count)

            if use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.ancestor_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[OMOP] CONCEPT_ANCESTOR 缓存已保存: {cache_path}")

    def _load_from_cache(self, cache_path: str):
        """从 pickle 缓存加载"""
        import time
        start = time.time()
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        self.standard = data['standard']
        self.classification = data['classification']
        self.valid = data['valid']
        self.standard_name_index = data['standard_name_index']
        self.standard_names = data['standard_names']
        self.classification_name_index = data.get('classification_name_index', {})
        self.classification_names = data.get('classification_names', [])
        self._build_domain_indexes()
        print(f"[OMOP] 缓存加载完成: {time.time() - start:.2f}s")
        print(f"[OMOP] 标准概念: {len(self.standard):,} 个, 分类概念: {len(self.classification):,} 个")

    def _save_to_cache(self, cache_path: str):
        """保存到 pickle 缓存"""
        import time
        start = time.time()
        data = {
            'standard': self.standard,
            'classification': self.classification,
            'valid': self.valid,
            'standard_name_index': self.standard_name_index,
            'standard_names': self.standard_names,
            'classification_name_index': self.classification_name_index,
            'classification_names': self.classification_names,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OMOP] 缓存已保存: {cache_path} ({time.time() - start:.2f}s)")

    def _build_indexes(self):
        """构建多种索引"""
        print("构建索引...")

        # 标准化名称
        self.standard['name_lower'] = self.standard['concept_name'].astype(str).str.lower().str.strip()
        self.valid['name_lower'] = self.valid['concept_name'].astype(str).str.lower().str.strip()
        self.classification['name_lower'] = self.classification['concept_name'].astype(str).str.lower().str.strip()

        # 标准概念名称索引
        self.standard_name_index = {}
        for _, row in self.standard.iterrows():
            name = row['name_lower']
            if pd.notna(name) and name:
                if name not in self.standard_name_index:
                    self.standard_name_index[name] = []
                self.standard_name_index[name].append({
                    'concept_id': int(row['concept_id']),
                    'concept_name': row['concept_name'],
                    'domain_id': row['domain_id'],
                    'vocabulary_id': row['vocabulary_id'],
                    'concept_class_id': row.get('concept_class_id', ''),
                })

        # 分类概念名称索引（ATC 类别等）
        self.classification_name_index = {}
        for _, row in self.classification.iterrows():
            name = row['name_lower']
            if pd.notna(name) and name:
                if name not in self.classification_name_index:
                    self.classification_name_index[name] = []
                self.classification_name_index[name].append({
                    'concept_id': int(row['concept_id']),
                    'concept_name': row['concept_name'],
                    'domain_id': row['domain_id'],
                    'vocabulary_id': row['vocabulary_id'],
                    'concept_class_id': row.get('concept_class_id', ''),
                })

        # 用于模糊匹配的名称列表
        self.standard_names = list(self.standard_name_index.keys())
        self.classification_names = list(self.classification_name_index.keys())
        self._build_domain_indexes()

        print(f"索引完成: {len(self.standard_name_index):,} 个标准概念名称, {len(self.classification_name_index):,} 个分类概念名称")

    def _build_domain_indexes(self):
        """Build per-domain indexes so fuzzy/contains matching does not scan all concepts."""
        self.standard_name_index_by_domain: Dict[str, Dict[str, list]] = {}
        for name, rows in self.standard_name_index.items():
            for row in rows:
                domain = row.get('domain_id')
                if not domain:
                    continue
                domain_index = self.standard_name_index_by_domain.setdefault(domain, {})
                domain_index.setdefault(name, []).append(row)

        self.standard_names_by_domain = {
            domain: list(index.keys())
            for domain, index in self.standard_name_index_by_domain.items()
        }

        self.classification_name_index_by_domain: Dict[str, Dict[str, list]] = {}
        for name, rows in self.classification_name_index.items():
            for row in rows:
                domain = row.get('domain_id')
                if not domain:
                    continue
                domain_index = self.classification_name_index_by_domain.setdefault(domain, {})
                domain_index.setdefault(name, []).append(row)

        self.classification_names_by_domain = {
            domain: list(index.keys())
            for domain, index in self.classification_name_index_by_domain.items()
        }

        self.standard_by_domain = {
            domain: df
            for domain, df in self.standard.groupby('domain_id', sort=False)
        }

    def _standard_index_for_domain(self, domain_id: Optional[str]):
        if domain_id is None:
            return self.standard_name_index
        return self.standard_name_index_by_domain.get(domain_id, {})

    def _standard_names_for_domain(self, domain_id: Optional[str]):
        if domain_id is None:
            return self.standard_names
        return self.standard_names_by_domain.get(domain_id, [])

    def _standard_rows_for_domain(self, domain_id: Optional[str]):
        if domain_id is None:
            return self.standard
        return self.standard_by_domain.get(domain_id, self.standard.iloc[0:0])

    def is_drug_class(self, concept_id: int) -> bool:
        """
        判断概念是否为药物类别（ATC 分类概念）

        Args:
            concept_id: 概念ID

        Returns:
            True 如果是药物类别（ATC），False 如果是具体药物
        """
        concept_info = self.get_concept_by_id(concept_id)
        if concept_info:
            return self.is_atc_class(concept_info)
        return False

    def is_drug_ingredient(self, concept_id: int) -> bool:
        """
        判断概念是否为具体药物成分（Ingredient）

        Args:
            concept_id: 概念ID

        Returns:
            True 如果是具体药物成分
        """
        concept_info = self.get_concept_by_id(concept_id)
        if concept_info:
            return self.is_ingredient(concept_info)
        return False

    def is_atc_class(self, concept_info: dict) -> bool:
        """
        判断是否为 ATC 药物类别概念

        Args:
            concept_info: 概念信息字典，包含 vocabulary_id 和 concept_class_id

        Returns:
            True 如果是 ATC 类别概念
        """
        return (
            concept_info.get('vocabulary_id') == 'ATC' or
            'ATC' in str(concept_info.get('concept_class_id', ''))
        )

    def is_ingredient(self, concept_info: dict) -> bool:
        """
        判断是否为药物成分（Ingredient）概念

        Args:
            concept_info: 概念信息字典

        Returns:
            True 如果是 Ingredient 概念
        """
        return concept_info.get('concept_class_id') == 'Ingredient'

    def _is_term_drug_class(self, term: str) -> bool:
        """
        判断术语是否可能是药物类别名称

        Args:
            term: 术语字符串

        Returns:
            True 如果术语可能是药物类别
        """
        term_lower = term.lower()

        # 检查是否包含类别关键词
        for keyword in self.DRUG_CLASS_KEYWORDS:
            if keyword in term_lower:
                return True

        # 检查是否以 's' 结尾（复数形式）
        words = term_lower.split()
        for word in words:
            if len(word) > 4 and word.endswith('s'):
                # 可能是复数形式的药物类别
                if any(kw in word for kw in ['inhibitor', 'antagonist', 'agonist', 'blocker']):
                    return True

        return False

    def _expand_term(self, term: str) -> List[tuple]:
        """
        扩展术语（缩写、同义词）

        Returns:
            [(扩展后的术语, 权重), ...]
        """
        expanded = [(term, 1.0)]
        term_upper = term.upper()
        term_lower = term.lower()

        # 1. 检查缩写扩展
        if term_upper in ABBREVIATION_DICT:
            expanded.append((ABBREVIATION_DICT[term_upper], 0.95))

        # 2. 检查中文映射
        if term_lower in CHINESE_ENGLISH_DICT:
            expanded.append((CHINESE_ENGLISH_DICT[term_lower], 0.95))

        # 3. 部分匹配中文词典
        for cn, en in CHINESE_ENGLISH_DICT.items():
            if cn in term_lower or term_lower in cn:
                expanded.append((en, 0.85))

        return expanded

    def match(self, english_term: str, domain_id: Optional[str] = None,
              top_k: int = 5) -> List[ConceptMatch]:
        """
        匹配概念

        Args:
            english_term: 英文术语
            domain_id: 可选的领域限制
            top_k: 返回结果数量上限

        Returns:
            匹配结果列表
        """
        results = []
        seen_ids = set()
        term_lower = english_term.lower().strip()
        standard_name_index = self._standard_index_for_domain(domain_id)
        standard_names = self._standard_names_for_domain(domain_id)
        standard_rows = self._standard_rows_for_domain(domain_id)

        # Step 1: 检查特殊映射
        for (key_term, key_domain), concept_id in SPECIAL_MAPPINGS.items():
            if key_term.lower() == term_lower:
                if key_domain is None or key_domain == domain_id:
                    row = self.valid[self.valid['concept_id'] == concept_id]
                    if not row.empty:
                        r = row.iloc[0]
                        return [ConceptMatch(
                            concept_id=int(r['concept_id']),
                            concept_name=r['concept_name'],
                            domain_id=r['domain_id'],
                            vocabulary_id=r['vocabulary_id'],
                            match_score=98,
                            match_type='special',
                            is_standard=r['standard_concept'] == 'S'
                        )]

        # 判断术语是否可能是药物类别
        is_drug_class_term = domain_id == 'Drug' and self._is_term_drug_class(english_term)

        # Step 2: 如果可能是药物类别，优先在分类概念中搜索
        if is_drug_class_term and domain_id == 'Drug':
            class_results = self._match_in_classification(english_term, domain_id, seen_ids)
            if class_results:
                # 找到 ATC 类别概念，直接返回
                return class_results[:top_k]

        # Step 3: 扩展术语并搜索
        expanded_terms = self._expand_term(english_term)

        for exp_term, weight in expanded_terms:
            exp_lower = exp_term.lower().strip()

            # 3.1 精确匹配
            if exp_lower in standard_name_index:
                for row in standard_name_index[exp_lower]:
                    if row['concept_id'] not in seen_ids:
                        seen_ids.add(row['concept_id'])
                        results.append(ConceptMatch(
                            concept_id=int(row['concept_id']),
                            concept_name=row['concept_name'],
                            domain_id=row['domain_id'],
                            vocabulary_id=row['vocabulary_id'],
                            match_score=100 * weight,
                            match_type='exact' if weight == 1.0 else 'synonym',
                            is_standard=True
                        ))
                if len(results) >= top_k and any(r.match_score >= 99.9 for r in results):
                    return sorted(results, key=lambda x: -x.match_score)[:top_k]

            # 3.2 包含匹配
            for name in standard_names:
                if exp_lower in name or name in exp_lower:
                    if exp_lower == name:
                        score = 100
                    elif exp_lower in name:
                        score = 90 * (len(exp_lower) / len(name))
                    else:
                        score = 85 * (len(name) / len(exp_lower))

                    for row in standard_name_index[name]:
                        if row['concept_id'] not in seen_ids:
                            seen_ids.add(row['concept_id'])
                            results.append(ConceptMatch(
                                concept_id=int(row['concept_id']),
                                concept_name=row['concept_name'],
                                domain_id=row['domain_id'],
                                vocabulary_id=row['vocabulary_id'],
                                match_score=score * weight,
                                match_type='contains',
                                is_standard=True
                            ))

        # Step 4: 模糊匹配（rapidfuzz 召回 + 词序列 SequenceMatcher 重排序，借鉴 LangExtract WordAligner）
        if len(results) < top_k:
            try:
                from rapidfuzz import fuzz, process
                fuzzy_results = process.extract(
                    term_lower,
                    standard_names,
                    scorer=fuzz.WRatio,
                    limit=_FUZZY_EXTRACT_LIMIT,
                )
                ranked: list[tuple[float, float, str]] = []
                for name, wratio, _ in fuzzy_results:
                    wratio_f = float(wratio)
                    combined = combined_fuzzy_score(term_lower, name, wratio_f)
                    if combined < _FUZZY_COMBINED_MIN and wratio_f < 70:
                        continue
                    eff_score = max(combined, wratio_f)
                    if eff_score > 100.0:
                        eff_score = 100.0
                    ranked.append((eff_score, wratio_f, name))
                ranked.sort(key=lambda x: (-x[0], -x[1]))

                for eff_score, _wratio, name in ranked:
                    for row in standard_name_index[name]:
                        if row['concept_id'] not in seen_ids:
                            seen_ids.add(row['concept_id'])
                            results.append(ConceptMatch(
                                concept_id=int(row['concept_id']),
                                concept_name=row['concept_name'],
                                domain_id=row['domain_id'],
                                vocabulary_id=row['vocabulary_id'],
                                match_score=eff_score,
                                match_type='fuzzy',
                                is_standard=True
                            ))
            except ImportError:
                pass

        # Step 5: 多关键词组合匹配
        keywords = re.findall(r'\b[a-z]{3,}\b', term_lower)
        if len(keywords) >= 2:
            for _, row in standard_rows.iterrows():
                name_lower = row['name_lower']
                if pd.isna(name_lower):
                    continue

                match_count = sum(1 for kw in keywords if kw in name_lower)
                if match_count >= len(keywords) * 0.5:
                    if row['concept_id'] not in seen_ids:
                        seen_ids.add(row['concept_id'])
                        score = 75 + (match_count / len(keywords)) * 15
                        results.append(ConceptMatch(
                            concept_id=int(row['concept_id']),
                            concept_name=row['concept_name'],
                            domain_id=row['domain_id'],
                            vocabulary_id=row['vocabulary_id'],
                            match_score=score,
                            match_type='multi_keyword',
                            is_standard=True
                        ))

        # Step 6: 如果术语是药物类别，过滤结果只保留类别概念
        if is_drug_class_term and results:
            class_results = [r for r in results if self.is_drug_class(r.concept_id)]
            if class_results:
                results = class_results

        # 去重并排序
        unique_results = []
        seen = set()
        for r in sorted(results, key=lambda x: -x.match_score):
            if r.concept_id not in seen:
                seen.add(r.concept_id)
                unique_results.append(r)

        return unique_results[:top_k]

    def _match_in_classification(self, english_term: str, domain_id: str,
                                  seen_ids: Set[int]) -> List[ConceptMatch]:
        """
        在分类概念（ATC 类别）中匹配

        Args:
            english_term: 英文术语
            domain_id: 领域ID
            seen_ids: 已见过的概念ID集合

        Returns:
            匹配结果列表
        """
        results = []
        term_lower = english_term.lower().strip()
        classification_name_index = self.classification_name_index_by_domain.get(
            domain_id,
            self.classification_name_index,
        )
        classification_names = self.classification_names_by_domain.get(
            domain_id,
            self.classification_names,
        )

        # 扩展术语
        expanded_terms = self._expand_term(english_term)

        for exp_term, weight in expanded_terms:
            exp_lower = exp_term.lower().strip()

            # 精确匹配
            if exp_lower in classification_name_index:
                for row in classification_name_index[exp_lower]:
                    if row['concept_id'] not in seen_ids:
                        seen_ids.add(row['concept_id'])
                        results.append(ConceptMatch(
                            concept_id=int(row['concept_id']),
                            concept_name=row['concept_name'],
                            domain_id=row['domain_id'],
                            vocabulary_id=row['vocabulary_id'],
                            match_score=100 * weight,
                            match_type='exact_class',
                            is_standard=False  # ATC 类别是 Classification 概念
                        ))

            # 包含匹配
            for name in classification_names:
                if exp_lower in name or name in exp_lower:
                    if exp_lower == name:
                        score = 100
                    elif exp_lower in name:
                        score = 90 * (len(exp_lower) / len(name))
                    else:
                        score = 85 * (len(name) / len(exp_lower))

                    for row in classification_name_index[name]:
                        if row['concept_id'] not in seen_ids:
                            seen_ids.add(row['concept_id'])
                            results.append(ConceptMatch(
                                concept_id=int(row['concept_id']),
                                concept_name=row['concept_name'],
                                domain_id=row['domain_id'],
                                vocabulary_id=row['vocabulary_id'],
                                match_score=score * weight,
                                match_type='contains_class',
                                is_standard=False
                            ))

        # 按分数排序
        results.sort(key=lambda x: -x.match_score)
        return results

    def get_concept_by_id(self, concept_id: int) -> Optional[dict]:
        """根据concept_id获取概念详情"""
        row = self.valid[self.valid['concept_id'] == concept_id]
        if not row.empty:
            r = row.iloc[0]
            return {
                'concept_id': int(r['concept_id']),
                'concept_name': r['concept_name'],
                'domain_id': r['domain_id'],
                'vocabulary_id': r['vocabulary_id'],
                'concept_class_id': r.get('concept_class_id', ''),
                'standard_concept': r.get('standard_concept', ''),
            }
        return None

    def get_descendants(self, concept_id: int) -> List[int]:
        """
        获取概念的所有后代概念ID

        Args:
            concept_id: 祖先概念ID

        Returns:
            后代概念ID列表
        """
        if self.ancestor_df is None:
            return []

        descendants = self.ancestor_df[
            (self.ancestor_df['ancestor_concept_id'] == concept_id) &
            (self.ancestor_df['descendant_concept_id'] != concept_id)  # 排除自身
        ]['descendant_concept_id'].tolist()

        return [int(d) for d in descendants]

    def get_ancestors(self, concept_id: int) -> List[int]:
        """
        获取概念的所有祖先概念ID

        Args:
            concept_id: 后代概念ID

        Returns:
            祖先概念ID列表
        """
        if self.ancestor_df is None:
            return []

        ancestors = self.ancestor_df[
            (self.ancestor_df['descendant_concept_id'] == concept_id) &
            (self.ancestor_df['ancestor_concept_id'] != concept_id)  # 排除自身
        ]['ancestor_concept_id'].tolist()

        return [int(a) for a in ancestors]
