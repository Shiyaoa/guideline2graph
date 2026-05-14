"""
分步测试新版 Pipeline - 模拟老版 Notebook 测试方式
使用真实 LLM API（通过 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 配置）
"""
import os
import sys
import asyncio
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.models import Provenance
from pipeline.lsh_cluster import lsh_cluster
from pipeline.config import set_config, PipelineConfig, LLMConfig, PathConfig
from pipeline.graph_api import (
    extract_terms_stage,
    extract_predicates_stage,
    extract_rules_stage,
    ClinicalGuidelinePipeline,
    save_results_from_cache
)
from pipeline.io_utils import save_clusters
import pandas as pd


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")


def require_llm_api_key() -> str:
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required for this real API manual test")
    return api_key


def load_provenances_from_excel(excel_path: str, max_per_sheet: int = None) -> list[Provenance]:
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


def setup_config(gen_dir: str):
    config = PipelineConfig(
        llm=LLMConfig(
            api_key=require_llm_api_key(),
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=16384,
            timeout=300.0
        ),
        paths=PathConfig(gen_dir=gen_dir)
    )
    set_config(config)
    return config


def step1_lsh_cluster(provenances: list[Provenance], gen_dir: str):
    print("\n" + "=" * 60)
    print("Step 1: LSH 聚类")
    print("=" * 60)

    start_time = datetime.now()
    result = lsh_cluster(provenances, max_cluster_size=8)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"[Step 1] 完成，耗时 {elapsed:.2f}s")
    print(f"[Step 1] 生成 {len(result.clusters)} 个聚类")

    sizes = [len(c.provenances) for c in result.clusters]
    print(f"[Step 1] 聚类大小: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

    cluster_path = os.path.join(gen_dir, "cluster.json")
    save_clusters(result.clusters, result.bucket_index, cluster_path)
    print(f"[Step 1] 已保存到 {cluster_path}")

    return result.clusters


def step2_extract_terms(clusters, gen_dir: str, max_concurrency: int = 5):
    print("\n" + "=" * 60)
    print("Step 2: 术语抽取")
    print("=" * 60)

    cluster_cache_path = os.path.join(gen_dir, "cluster_cache.json")

    start_time = datetime.now()
    result = extract_terms_stage(
        clusters=clusters,
        max_concurrency=max_concurrency,
        persist_cluster_cache=True,
        cluster_cache_path=cluster_cache_path
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    terms = result.get("terms", [])
    med_terms = result.get("med_terms", [])

    print(f"[Step 2] 完成，耗时 {elapsed:.2f}s")
    print(f"[Step 2] 术语: {len(terms)} 个")
    print(f"[Step 2] 药物: {len(med_terms)} 个")

    if terms:
        print("\n  术语示例:")
        for t in terms[:5]:
            print(f"    - {t.name} ({t.label})")

    if med_terms:
        print("\n  药物示例:")
        for m in med_terms[:5]:
            print(f"    - {m.name}")

    return result


def step3_extract_predicates(clusters, gen_dir: str, max_concurrency: int = 5):
    print("\n" + "=" * 60)
    print("Step 3: 谓词抽取")
    print("=" * 60)

    cluster_cache_path = os.path.join(gen_dir, "cluster_cache.json")

    start_time = datetime.now()
    result = extract_predicates_stage(
        clusters=clusters,
        max_concurrency=max_concurrency,
        persist_cluster_cache=True,
        cluster_cache_path=cluster_cache_path
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    predicates = result.get("predicates", [])

    print(f"[Step 3] 完成，耗时 {elapsed:.2f}s")
    print(f"[Step 3] 谓词: {len(predicates)} 个")

    if predicates:
        print("\n  谓词示例:")
        for p in predicates[:5]:
            preview = (p.description or p.source_text or p.id)[:60]
            print(f"    - {p.name}: {p.input_shape} -> {p.final_output_type}; {preview}...")

    return result


def step4_extract_rules(clusters, gen_dir: str, max_concurrency: int = 5):
    print("\n" + "=" * 60)
    print("Step 4: 规则抽取")
    print("=" * 60)

    cluster_cache_path = os.path.join(gen_dir, "cluster_cache.json")

    start_time = datetime.now()
    result = extract_rules_stage(
        clusters=clusters,
        max_concurrency=max_concurrency,
        persist_cluster_cache=True,
        cluster_cache_path=cluster_cache_path
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    rules = result.get("rules", [])

    print(f"[Step 4] 完成，耗时 {elapsed:.2f}s")
    print(f"[Step 4] 规则: {len(rules)} 条")

    if rules:
        print("\n  规则示例:")
        for r in rules[:5]:
            subjects = ", ".join(r.action.subjects) if r.action.subjects else "N/A"
            print(f"    - {r.label}: {r.action.permission.value} {subjects}")

    return result


def step5_save_results(gen_dir: str):
    print("\n" + "=" * 60)
    print("Step 5: 保存最终结果")
    print("=" * 60)

    cluster_cache_path = os.path.join(gen_dir, "cluster_cache.json")

    save_results_from_cache(
        cluster_cache_path=cluster_cache_path,
        gen_dir=gen_dir,
        save_provenances=False
    )

    print(f"[Step 5] 结果已保存到 {gen_dir}/")


async def run_async_pipeline(clusters, gen_dir: str):
    print("\n" + "=" * 60)
    print("异步测试: process_clusters_async")
    print("=" * 60)

    pipeline = ClinicalGuidelinePipeline()
    cluster_cache_path = os.path.join(gen_dir, "cluster_cache_async.json")

    start_time = datetime.now()
    result = await pipeline.process_clusters_async(
        clusters=clusters,
        cluster_cache_path=cluster_cache_path,
        persist_cluster_cache=True,
        verbose=True
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n[异步测试] 完成，耗时 {elapsed:.2f}s")
    print(f"[异步测试] 结果: { {k: len(v) for k, v in result.items()} }")

    return result


def main():
    gen_dir = "gen_test"
    os.makedirs(gen_dir, exist_ok=True)
    setup_config(gen_dir)

    excel_path = "SGLT2i_zh.xlsx"
    if not os.path.exists(excel_path):
        print(f"错误: 找不到 {excel_path}")
        return

    provenances = load_provenances_from_excel(excel_path, max_per_sheet=5)

    clusters = step1_lsh_cluster(provenances, gen_dir)
    step2_extract_terms(clusters, gen_dir, max_concurrency=3)
    step3_extract_predicates(clusters, gen_dir, max_concurrency=3)
    step4_extract_rules(clusters, gen_dir, max_concurrency=3)
    step5_save_results(gen_dir)

    print("\n" + "=" * 60)
    print("分步测试完成!")
    print("=" * 60)


async def main_async():
    gen_dir = "gen_test_async"
    os.makedirs(gen_dir, exist_ok=True)
    setup_config(gen_dir)

    excel_path = "SGLT2i_zh.xlsx"
    if not os.path.exists(excel_path):
        print(f"错误: 找不到 {excel_path}")
        return

    provenances = load_provenances_from_excel(excel_path, max_per_sheet=5)
    clusters = step1_lsh_cluster(provenances, gen_dir)
    await run_async_pipeline(clusters, gen_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", action="store_true", help="使用异步模式测试")
    args = parser.parse_args()

    if getattr(args, "async"):
        asyncio.run(main_async())
    else:
        main()
