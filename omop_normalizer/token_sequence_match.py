"""
词序列对齐评分（借鉴 LangExtract WordAligner 思路）

对英文查询与候选 OMOP 概念名在「词 token」层面使用 difflib.SequenceMatcher，
对词序变化、增删词比纯字符级模糊匹配更稳；与 rapidfuzz 分数融合用于重排序。
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache

# 与 LangExtract resolver 中 _normalize_token 类似的轻量归一化，减轻复数等形态差异
@lru_cache(maxsize=32768)
def normalize_token_light(token: str) -> str:
    t = token.lower()
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def english_word_tokens(text: str) -> list[str]:
    """从英文术语中拆出词级 token（小写 + 轻量归一化）。"""
    return [normalize_token_light(m.group(0)) for m in _WORD_RE.finditer(text)]


def token_sequence_ratio(query: str, candidate: str) -> float:
    """
    词序列相似度，范围约 0–100。

    使用 SequenceMatcher(autojunk=False) 在 token 列表上比对，与 LangExtract
    WordAligner 中基于 difflib 的对齐一致思想，适用于术语与概念名的对齐评分。
    """
    a = english_word_tokens(query)
    b = english_word_tokens(candidate)
    if not a or not b:
        return 0.0
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return matcher.ratio() * 100.0


# rapidfuzz WRatio 与词序列分融合；词序列略加权，利于「词序重排」类 OMOP 名称
_DEFAULT_WRATIO_WEIGHT = 0.38
_DEFAULT_TOKEN_WEIGHT = 0.62


def combined_fuzzy_score(
    query: str,
    candidate_name: str,
    wratio: float,
    *,
    wratio_weight: float = _DEFAULT_WRATIO_WEIGHT,
    token_weight: float = _DEFAULT_TOKEN_WEIGHT,
) -> float:
    """融合 rapidfuzz 分数与词序列分，用于模糊候选重排序（权重之和应为 1）。"""
    ts = token_sequence_ratio(query, candidate_name)
    return wratio_weight * float(wratio) + token_weight * ts
