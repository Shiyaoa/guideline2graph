"""
后处理标准化模块

Pipeline 完成后执行：
1. 加载 terms.json 和 med_terms.json
2. 执行 OMOP 匹配，生成映射表
3. 根据映射表合并同义术语
4. 更新 predicates 和 rules 中的引用
"""
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .term_mapping import TermMappingRegistry, get_registry
from .processors import CombinedTermProcessor
from .models import Term, MedicationTerm
from .binding_resolver import resolve_bindings_for_terms


def load_json(filepath: str) -> List[dict]:
    """加载 JSON 文件"""
    if not Path(filepath).exists():
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath: str) -> None:
    """保存 JSON 文件"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_id_replacement_map(registry: TermMappingRegistry) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    构建 ID 替换映射表

    对于每个同义术语组，选择 match_score 最高的作为主 ID，
    其他 ID 映射到主 ID。

    Returns:
        (term_replacements, med_replacements) 两个替换映射
    """
    term_replacements = {}
    med_replacements = {}

    # 处理非药物术语
    for concept_id, semantic_ids in registry.get_all_term_groups().items():
        if len(semantic_ids) <= 1:
            continue

        # 选择 match_score 最高的作为主 ID
        primary = registry.get_primary_term_id(semantic_ids[0])
        for sid in semantic_ids:
            if sid != primary:
                term_replacements[sid] = primary

    # 处理药物术语
    for concept_id, semantic_ids in registry.get_all_med_groups().items():
        if len(semantic_ids) <= 1:
            continue

        primary = registry.get_primary_med_id(semantic_ids[0])
        for sid in semantic_ids:
            if sid != primary:
                med_replacements[sid] = primary

    return term_replacements, med_replacements


def update_predicate_references(predicates: List[dict],
                                 term_replacements: Dict[str, str],
                                 med_replacements: Dict[str, str]) -> List[dict]:
    """
    更新谓词中的术语引用

    Args:
        predicates: 谓词列表
        term_replacements: 术语 ID 替换映射
        med_replacements: 药物 ID 替换映射

    Returns:
        更新后的谓词列表
    """
    all_replacements = {**term_replacements, **med_replacements}
    updated = []

    for pred in predicates:
        new_pred = _replace_nested(pred.copy(), all_replacements)

        # 更新 dependencies
        if "dependencies" in new_pred:
            new_pred["dependencies"] = [
                all_replacements.get(dep, dep)
                for dep in new_pred["dependencies"]
            ]

        # 更新 id（谓词 ID 包含术语 ID）
        if "id" in new_pred:
            new_pred["id"] = _replace_in_predicate_id(new_pred["id"], all_replacements)

        # legacy-only: 更新 formal_definition（可能包含术语引用）
        if "formal_definition" in new_pred:
            new_pred["formal_definition"] = _replace_in_text(
                new_pred["formal_definition"], all_replacements
            )

        updated.append(new_pred)

    return updated


def update_rule_references(rules: List[dict],
                           term_replacements: Dict[str, str],
                           med_replacements: Dict[str, str]) -> List[dict]:
    """
    更新规则中的术语引用

    Args:
        rules: 规则列表
        term_replacements: 术语 ID 替换映射
        med_replacements: 药物 ID 替换映射

    Returns:
        更新后的规则列表
    """
    all_replacements = {**term_replacements, **med_replacements}
    updated = []

    for rule in rules:
        new_rule = _replace_nested(rule.copy(), all_replacements)

        # legacy-only: 更新 condition（布尔表达式，包含谓词 ID）
        if "condition" in new_rule:
            new_rule["condition"] = _replace_in_text(
                new_rule["condition"], all_replacements
            )

        # 更新 action.subjects（药物 ID 列表）
        if "action" in new_rule and "subjects" in new_rule["action"]:
            new_rule["action"] = new_rule["action"].copy()
            new_rule["action"]["subjects"] = [
                med_replacements.get(s, s)
                for s in new_rule["action"]["subjects"]
            ]

        # 更新 id
        if "id" in new_rule:
            new_rule["id"] = _replace_in_predicate_id(new_rule["id"], all_replacements)

        updated.append(new_rule)

    return updated


def _replace_nested(value: Any, replacements: Dict[str, str]) -> Any:
    """递归替换 v2 schema 中嵌套字段里的 term/med id。"""
    if isinstance(value, str):
        return _replace_in_text(value, replacements)
    if isinstance(value, list):
        return [_replace_nested(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: _replace_nested(v, replacements) for k, v in value.items()}
    return value


def merge_terms(terms: List[dict], replacements: Dict[str, str]) -> List[dict]:
    """
    合并同义术语

    Args:
        terms: 术语列表
        replacements: ID 替换映射（旧 ID -> 主 ID）

    Returns:
        合并后的术语列表
    """
    # 收集被合并的术语信息
    merged_info: Dict[str, List[str]] = {}  # primary_id -> [aliases]

    for term in terms:
        old_id = term["id"]
        if old_id in replacements:
            primary_id = replacements[old_id]
            if primary_id not in merged_info:
                merged_info[primary_id] = []
            merged_info[primary_id].append(old_id)

    # 过滤掉被合并的术语，添加 aliases 字段
    merged_terms = []
    primary_ids = set(replacements.values())

    for term in terms:
        term_id = term["id"]

        # 跳过被合并的术语
        if term_id in replacements:
            continue

        new_term = term.copy()

        # 添加 aliases 字段
        if term_id in merged_info:
            new_term["aliases"] = merged_info[term_id]

        merged_terms.append(new_term)

    return merged_terms


def _replace_in_predicate_id(pred_id: str, replacements: Dict[str, str]) -> str:
    """
    替换谓词 ID 中的术语引用

    v2 谓词 ID 示例: pred.meas.egfr.value.lt.30
    legacy 谓词 ID 也会做 best-effort 替换。
    """
    parts = pred_id.split(".")
    if len(parts) < 3:
        return pred_id

    # 从第 2 个部分开始检查（跳过 "pred" 和 operator）
    for i in range(2, len(parts)):
        # 尝试匹配术语 ID（可能跨越多个部分）
        for old_id, new_id in replacements.items():
            old_parts = old_id.split(".")
            if parts[i:i+len(old_parts)] == old_parts:
                parts = parts[:i] + new_id.split(".") + parts[i+len(old_parts):]
                break

    return ".".join(parts)


def _replace_in_text(text: str, replacements: Dict[str, str]) -> str:
    """
    替换文本中的术语 ID 引用

    Args:
        text: 包含术语 ID 的文本
        replacements: 替换映射

    Returns:
        替换后的文本
    """
    result = text
    # 按长度降序排列，避免部分匹配
    for old_id, new_id in sorted(replacements.items(), key=lambda x: -len(x[0])):
        result = result.replace(old_id, new_id)
    return result


def _perform_omop_matching(
    terms: List[dict],
    med_terms: List[dict],
    enable_review: bool = True
) -> None:
    """
    执行 OMOP 匹配，注册映射到 registry

    Args:
        terms: 术语列表（dict 格式）
        med_terms: 药物术语列表（dict 格式）
        enable_review: 是否启用 LLM 审核
    """
    from .config import get_config

    registry = get_registry()
    processor = CombinedTermProcessor(
        config=get_config().match,
        enable_cache=True,
        enable_review=enable_review
    )

    # 转换为 Term/MedicationTerm 对象
    term_objs = [Term(**t) for t in terms if isinstance(t, dict)]
    med_term_objs = [MedicationTerm(**m) for m in med_terms if isinstance(m, dict)]

    from .models import TermExtractionResult
    result = TermExtractionResult(terms=term_objs, med_terms=med_term_objs)

    # 执行 OMOP 匹配（注册到 registry，但不修改 Term）
    processor.process(result)

    print(f"[OMOP匹配] 术语映射: {registry.term_count} 条")
    print(f"[OMOP匹配] 药物映射: {registry.med_count} 条")
    print(f"[OMOP匹配] 同义术语组: {registry.synonym_group_count} 组")


def standardize_terms(
    terms_path: str = "gen/terms.json",
    med_terms_path: str = "gen/med_terms.json",
    predicates_path: str = "gen/predicates.json",
    rules_path: str = "gen/rules.json",
    mapping_path: str = "gen/term_omop_mapping.json",
    output_dir: str = "gen",
    enable_review: bool = True,
    skip_omop_matching: bool = False
) -> Dict:
    """
    后处理标准化：执行或复用 OMOP 匹配，并合并同义术语

    流程：
    1. 加载数据文件（terms.json, med_terms.json）
    2. 执行 OMOP 匹配生成映射表，或在调试模式下跳过并复用已有映射表
    3. 保存映射表
    4. 构建替换映射，合并同义术语
    5. 更新 predicates 和 rules 中的引用
    6. 输出最终标准化结果

    Args:
        terms_path: 术语文件路径
        med_terms_path: 药物术语文件路径
        predicates_path: 谓词文件路径
        rules_path: 规则文件路径
        mapping_path: 映射表文件路径
        output_dir: 输出目录
        enable_review: 是否启用 LLM 审核 OMOP 匹配
        skip_omop_matching: 是否跳过 OMOP 匹配（仅加载已有映射表）

    Returns:
        标准化统计信息
    """
    # 1. 加载数据文件
    terms = load_json(terms_path) if Path(terms_path).exists() else []
    med_terms = load_json(med_terms_path) if Path(med_terms_path).exists() else []
    predicates = load_json(predicates_path) if Path(predicates_path).exists() else []
    rules = load_json(rules_path) if Path(rules_path).exists() else []

    print(f"[标准化] 加载术语: {len(terms)} 个")
    print(f"[标准化] 加载药物: {len(med_terms)} 个")

    # 2. 初始化 registry
    registry = get_registry()
    registry.clear()

    # 3. 执行 OMOP 匹配或加载已有映射表
    if skip_omop_matching:
        print("[标准化] 跳过 OMOP 匹配，加载已有映射表...")
        registry.load(mapping_path)
    else:
        print("[标准化] 执行 OMOP 匹配...")
        _perform_omop_matching(terms, med_terms, enable_review=enable_review)

        # 保存映射表
        registry.save(mapping_path)
        print(f"[标准化] 映射表已保存: {mapping_path}")

    # 4. 构建替换映射
    term_replacements, med_replacements = build_id_replacement_map(registry)

    print(f"[标准化] 术语替换映射: {len(term_replacements)} 条")
    print(f"[标准化] 药物替换映射: {len(med_replacements)} 条")

    # 5. 解析 binding：FHIR resource/path 来自本地 registry；OMOP 只生成
    # candidate terminology code，不直接升级为 verified ValueSet。
    terms, med_terms = resolve_bindings_for_terms(terms, med_terms, registry)

    # 6. 更新引用
    updated_predicates = update_predicate_references(
        predicates, term_replacements, med_replacements
    )
    updated_rules = update_rule_references(
        rules, term_replacements, med_replacements
    )

    # 7. 合并术语
    merged_terms = merge_terms(terms, term_replacements)
    merged_med_terms = merge_terms(med_terms, med_replacements)

    print(f"[标准化] 术语: {len(terms)} -> {len(merged_terms)}")
    print(f"[标准化] 药物: {len(med_terms)} -> {len(merged_med_terms)}")

    # 8. 输出标准化结果
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    save_json(merged_terms, f"{output_dir}/terms_standardized.json")
    save_json(merged_med_terms, f"{output_dir}/med_terms_standardized.json")
    save_json(updated_predicates, f"{output_dir}/predicates_standardized.json")
    save_json(updated_rules, f"{output_dir}/rules_standardized.json")

    return {
        "omop_matching_skipped": skip_omop_matching,
        "term_replacements": len(term_replacements),
        "med_replacements": len(med_replacements),
        "term_mappings": registry.term_count,
        "med_mappings": registry.med_count,
        "original_term_count": len(terms),
        "final_term_count": len(merged_terms),
        "original_med_count": len(med_terms),
        "final_med_count": len(merged_med_terms),
        "synonym_groups": len(term_replacements) + len(med_replacements)
    }


# 便捷函数
def run_standardization(
    gen_dir: str = "gen",
    enable_review: bool = True,
    skip_omop_matching: bool = False,
) -> Dict:
    """
    运行标准化流程（便捷入口）

    Args:
        gen_dir: 输出目录
        enable_review: 是否启用 LLM 审核 OMOP 匹配
        skip_omop_matching: 是否跳过本地 OMOP 匹配，仅复用已有 term_omop_mapping.json。
            适合复用 cluster cache 调试 term/predicate/rule 抽取链路。

    Returns:
        标准化统计信息
    """
    return standardize_terms(
        terms_path=f"{gen_dir}/terms.json",
        med_terms_path=f"{gen_dir}/med_terms.json",
        predicates_path=f"{gen_dir}/predicates.json",
        rules_path=f"{gen_dir}/rules.json",
        mapping_path=f"{gen_dir}/term_omop_mapping.json",
        output_dir=gen_dir,
        enable_review=enable_review,
        skip_omop_matching=skip_omop_matching,
    )
