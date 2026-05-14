"""
运行术语抽取测试并保存结果
使用真实 LLM API（通过 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 配置）
"""
import os
import json
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.models import Provenance
from pipeline.lsh_cluster import lsh_cluster
from pipeline.config import set_config, PipelineConfig, LLMConfig, PathConfig
from pipeline.graph_api import extract_terms_stage
import pandas as pd


def require_llm_api_key() -> str:
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required for this real API manual test")
    return api_key


def load_provenances_from_excel(excel_path: str, max_per_sheet: int = None):
    """从 Excel 加载推荐意见"""
    xl = pd.ExcelFile(excel_path)
    provenances = []

    df_guideline = pd.read_excel(xl, sheet_name=1)
    source_col = df_guideline.columns[0]
    quote_col = df_guideline.columns[1]

    count = 0
    for idx, row in df_guideline.iterrows():
        if max_per_sheet and count >= max_per_sheet:
            break
        if pd.notna(row[quote_col]):
            provenances.append(Provenance(
                source=str(row[source_col]) if pd.notna(row[source_col]) else "指南",
                quote=str(row[quote_col]),
                type="guideline"
            ))
            count += 1

    df_consensus = pd.read_excel(xl, sheet_name=2)
    source_col = df_consensus.columns[0]
    quote_col = df_consensus.columns[1]

    count = 0
    for idx, row in df_consensus.iterrows():
        if max_per_sheet and count >= max_per_sheet:
            break
        if pd.notna(row[quote_col]):
            provenances.append(Provenance(
                source=str(row[source_col]) if pd.notna(row[source_col]) else "专家共识",
                quote=str(row[quote_col]),
                type="expert"
            ))
            count += 1

    print(f"[加载] 共加载 {len(provenances)} 条推荐意见")
    return provenances


def main():
    gen_dir = "gen"
    os.makedirs(gen_dir, exist_ok=True)

    config = PipelineConfig(
        llm=LLMConfig(
            api_key=require_llm_api_key(),
            base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("LLM_MODEL", "deepseek-v4-pro")
        ),
        paths=PathConfig(gen_dir=gen_dir)
    )
    set_config(config)

    excel_path = "SGLT2i_zh.xlsx"
    if not os.path.exists(excel_path):
        print(f"错误: 找不到 {excel_path}")
        return

    provenances = load_provenances_from_excel(excel_path, max_per_sheet=3)

    print("\n[Step 1] LSH 聚类...")
    start_time = datetime.now()
    lsh_result = lsh_cluster(provenances, max_cluster_size=5)
    cluster_time = (datetime.now() - start_time).total_seconds()
    print(f"[Step 1] 完成，耗时 {cluster_time:.2f}s，生成 {len(lsh_result.clusters)} 个聚类")

    print("\n[Step 2] 术语抽取...")
    start_time = datetime.now()
    result = extract_terms_stage(
        clusters=lsh_result.clusters,
        max_concurrency=1,
        persist_cluster_cache=False
    )
    extract_time = (datetime.now() - start_time).total_seconds()
    print(f"[Step 2] 完成，耗时 {extract_time:.2f}s")

    terms = result.get("terms", [])
    med_terms = result.get("med_terms", [])

    print(f"\n[结果汇总]")
    print(f"  术语: {len(terms)} 个")
    print(f"  药物: {len(med_terms)} 个")

    if terms:
        print("\n  术语列表:")
        for t in terms:
            print(f"    - {t.id}: {t.name} ({t.label})")

    if med_terms:
        print("\n  药物列表:")
        for m in med_terms:
            print(f"    - {m.id}: {m.name}")

    output = {
        "metadata": {
            "total_provenances": len(provenances),
            "total_clusters": len(lsh_result.clusters),
            "cluster_time": cluster_time,
            "extract_time": extract_time,
            "total_time": cluster_time + extract_time
        },
        "terms": [t.model_dump() for t in terms],
        "med_terms": [m.model_dump() for m in med_terms]
    }

    output_path = os.path.join(gen_dir, "test_terms_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
