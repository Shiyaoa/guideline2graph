# -*- coding: utf-8 -*-
"""
完整 Pipeline 测试：使用 SGLT2i_zh.xlsx 的「指南」和「专家共识」两个 sheet
Phase 1: 推荐意见抽取 → LSH 聚类 → 术语/谓词/规则抽取（异步）
Phase 2: OMOP 标准化
记录每阶段用时

用法示例：
  python run_full_test.py --excel "D:\\data\\SGLT2i_zh.xlsx" --run-eval
  python run_full_test.py --excel test.xlsx --limit 20 --skip-omop-matching
  或设置环境变量 SGLT2I_EXCEL 后省略 --excel（默认当前目录下 SGLT2i_zh.xlsx）。
  API Key：LLM_API_KEY 或 DASHSCOPE_API_KEY（二选一）。
"""
import argparse
import os
import sys
import asyncio
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR)

try:
    from dotenv import load_dotenv

    load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    pass

# Windows 终端下避免长时间无输出：行缓冲 + 每次 print 刷新
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    except Exception:
        pass


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


SHEET_GUIDELINE = "指南"
SHEET_EXPERT = "专家共识"
TEXT_COL = "推荐意见"
SOURCE_COL = "来源"
MAX_CONCURRENCY = 8   # 异步并发数


def _resolve_api_key() -> str:
    return (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()


def _resolve_base_url() -> str:
    return os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()


def _resolve_model() -> str:
    return os.getenv("LLM_MODEL", "deepseek-v4-pro").strip()


def _resolve_excel_path(cli_excel: str) -> Path:
    raw = (cli_excel or os.getenv("SGLT2I_EXCEL") or "SGLT2i_zh.xlsx").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = _SCRIPT_DIR / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGLT2i 完整 pipeline（指南+专家共识）+ 可选 eval")
    p.add_argument(
        "--excel",
        default=os.getenv("SGLT2I_EXCEL", "SGLT2i_zh.xlsx"),
        help="SGLT2i_zh.xlsx 路径；默认同目录或环境变量 SGLT2I_EXCEL",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help="术语/谓词/规则异步阶段并发数",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多读取多少条推荐意见；适合 test.xlsx 小批量 smoke run",
    )
    p.add_argument(
        "--skip-standardization",
        action="store_true",
        help="跳过 Phase 2 OMOP 标准化",
    )
    p.add_argument(
        "--skip-omop-matching",
        action="store_true",
        help="运行标准化输出，但跳过慢速本地 OMOP matching，仅复用已有 term_omop_mapping.json",
    )
    p.add_argument(
        "--run-eval",
        action="store_true",
        help="pipeline 结束后：生成 gold 模板并运行 source-grounded eval",
    )
    p.add_argument(
        "--eval-dry-run",
        action="store_true",
        help="与 --run-eval 联用：仅构建上下文，不调用 judge LLM",
    )
    p.add_argument(
        "--eval-max-items",
        type=int,
        default=None,
        help="eval 最多评估条数（默认全部）",
    )
    return p.parse_args()


# ──────────────────────────────────────────────
# 工具：用时记录
# ──────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._records: list[tuple[str, float]] = []
        self._start = time.time()
        self._lap = self._start

    def lap(self, label: str) -> float:
        now = time.time()
        elapsed = now - self._lap
        self._records.append((label, elapsed))
        self._lap = now
        _print(f"  [timing] {label}: {elapsed:.1f}s")
        return elapsed

    def total(self) -> float:
        return time.time() - self._start

    def summary(self) -> dict:
        return {
            "steps": [{"step": l, "seconds": round(s, 1)} for l, s in self._records],
            "total_seconds": round(self.total(), 1),
        }


# ──────────────────────────────────────────────
# 读取 Excel
# ──────────────────────────────────────────────

def load_texts(excel_path: Path, limit: int | None = None) -> tuple[list[str], list[str]]:
    """返回 (texts, sources)"""
    if not excel_path.is_file():
        raise FileNotFoundError(f"未找到 Excel 文件: {excel_path}")
    df_all = pd.read_excel(excel_path, sheet_name=None)
    target_sheets = {
        name: df
        for name, df in df_all.items()
        if name in (SHEET_GUIDELINE, SHEET_EXPERT)
    }
    if not target_sheets:
        target_sheets = df_all

    records: list[dict] = []
    for sheet_name, df in target_sheets.items():
        if sheet_name == SHEET_GUIDELINE:
            src_type = "guideline"
        elif sheet_name == SHEET_EXPERT:
            src_type = "expert"
        else:
            src_type = sheet_name
        for _, row in df.iterrows():
            text = str(row.get(TEXT_COL, "")).strip()
            source = str(row.get(SOURCE_COL, "")).strip()
            if text and text != "nan":
                records.append({"text": text, "source": source, "type": src_type})
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break

    _print(f"  共加载 {len(records)} 条推荐意见")
    sheet_counts = {}
    for r in records:
        sheet_counts[r["type"]] = sheet_counts.get(r["type"], 0) + 1
    for k, v in sheet_counts.items():
        _print(f"    - {k}: {v} 条")

    texts = [r["text"] for r in records]
    sources = [r["source"] for r in records]
    return texts, sources


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

async def run_async_extraction(clusters, cluster_cache_path, gen_dir, max_concurrency: int):
    from pipeline import process_clusters_async
    return await process_clusters_async(
        clusters=clusters,
        cluster_cache_path=cluster_cache_path,
        persist_cluster_cache=True,
        max_concurrency=max_concurrency,
        verbose=True,
        save_results=True,
        gen_dir=gen_dir,
    )


def _run_source_grounded_eval(gen_dir: str, dry_run: bool, max_items: int | None) -> None:
    """在 gen_dir 上生成 gold 模板并运行 eval（gold 为占位，适合 smoke / 人工后续补标）。"""
    gold_path = os.path.join(gen_dir, "gold_template.jsonl")
    eval_py = _SCRIPT_DIR / "scripts" / "eval" / "run_source_grounded_eval.py"
    env = {**os.environ, "PYTHONPATH": str(_SCRIPT_DIR)}

    cmd_write = [
        sys.executable,
        str(eval_py),
        "--gen-dir",
        gen_dir,
        "--write-gold-template",
        gold_path,
    ]
    if max_items is not None:
        cmd_write.extend(["--max-items", str(max_items)])
    _print("\n[Eval] 生成 gold 模板...")
    r0 = subprocess.run(cmd_write, cwd=str(_SCRIPT_DIR), env=env)
    if r0.returncode != 0:
        raise RuntimeError("write-gold-template 失败")

    out_dir = os.path.join(gen_dir, "eval_source_grounded")
    cmd_eval = [
        sys.executable,
        str(eval_py),
        "--gen-dir",
        gen_dir,
        "--gold",
        gold_path,
        "--out-dir",
        out_dir,
    ]
    if dry_run:
        cmd_eval.append("--dry-run")
    if max_items is not None:
        cmd_eval.extend(["--max-items", str(max_items)])
    _print("[Eval] 运行 source-grounded eval...")
    r1 = subprocess.run(cmd_eval, cwd=str(_SCRIPT_DIR), env=env)
    if r1.returncode != 0:
        raise RuntimeError("source-grounded eval 失败")
    _print(f"[Eval] 完成，输出目录: {out_dir}")


def main():
    args = parse_args()
    excel_path = _resolve_excel_path(args.excel)
    api_key = _resolve_api_key()
    base_url = _resolve_base_url()
    model = _resolve_model()
    if not api_key:
        _print("请设置环境变量 LLM_API_KEY 或 DASHSCOPE_API_KEY（可在 guideline2graph/.env 中配置）")
        sys.exit(1)

    timer = Timer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_dir = f"gen/full_test_{timestamp}"
    os.makedirs(gen_dir, exist_ok=True)
    cluster_cache_path = os.path.join(gen_dir, "cluster_cache.json")

    _print("=" * 65)
    _print(f"SGLT2i 完整 Pipeline 测试  [{timestamp}]")
    _print(f"Excel: {excel_path}")
    _print(f"输出目录: {gen_dir}")
    _print("=" * 65)

    # ── 读取数据 ──────────────────────────────
    _print("\n[Step 0] 读取 Excel 数据...")
    texts, _ = load_texts(excel_path, limit=args.limit)
    timer.lap("读取 Excel")

    # ── 初始化 Pipeline ───────────────────────
    preload_omop = not (args.skip_standardization or args.skip_omop_matching)
    preload_note = "预加载 OMOP" if preload_omop else "不预加载 OMOP"
    _print(f"\n[Step 0] 初始化 Pipeline ({model}, {preload_note})...")
    _print(f"LLM base_url: {base_url}")
    from pipeline import (
        create_pipeline,
        extract_provenances_stage,
        cluster_provenances_stage,
        save_results_from_cache,
        run_standardization,
        get_failed_task_logger,
    )

    get_failed_task_logger().reset()
    pipeline = create_pipeline(
        api_key=api_key,
        base_url=base_url,
        model=model,
        preload_omop=preload_omop,
        gen_dir=gen_dir,
    )
    get_failed_task_logger().reset()
    timer.lap("初始化 Pipeline + OMOP 预加载")

    # ══ PHASE 1 ══════════════════════════════

    # ── Step 1: Provenance 抽取 ───────────────
    _print(f"\n[Step 1] 推荐意见抽取 ({len(texts)} 条, max_concurrency=5)...")
    provenances_path = os.path.join(gen_dir, "provenances.json")
    provenances = extract_provenances_stage(
        texts,
        save_to_file=True,
        filepath=provenances_path,
        max_concurrency=5,
    )
    _print(f"  抽取到 {len(provenances)} 条推荐意见")
    timer.lap("Step 1 - Provenance 抽取")

    # ── Step 2: LSH 聚类 ──────────────────────
    _print(f"\n[Step 2] LSH 聚类...")
    clusters_path = os.path.join(gen_dir, "clusters.json")
    clusters, bucket_index = cluster_provenances_stage(
        provenances=provenances,
        save_to_file=True,
        filepath=clusters_path,
    )
    _print(f"  {len(provenances)} 条 → {len(clusters)} 个聚类")
    timer.lap("Step 2 - LSH 聚类")

    # ── Step 3: 异步抽取（术语 + 谓词 + 规则）──
    _print(f"\n[Step 3] 异步抽取 术语/谓词/规则 (max_concurrency={args.max_concurrency})...")
    extraction_result = asyncio.run(
        run_async_extraction(clusters, cluster_cache_path, gen_dir, args.max_concurrency)
    )
    _print(f"  Terms:      {len(extraction_result.get('terms', []))}")
    _print(f"  Med Terms:  {len(extraction_result.get('med_terms', []))}")
    _print(f"  Predicates: {len(extraction_result.get('predicates', []))}")
    _print(f"  Rules:      {len(extraction_result.get('rules', []))}")
    timer.lap("Step 3 - 异步知识抽取 (术语+谓词+规则)")

    # ── Step 4: 汇总保存 ──────────────────────
    _print(f"\n[Step 4] 汇总保存...")
    save_results_from_cache(
        cluster_cache_path=cluster_cache_path,
        gen_dir=gen_dir,
    )
    timer.lap("Step 4 - 汇总保存 cluster_final.json")

    # ── Phase 1 失败统计 ──────────────────────
    failed_count = get_failed_task_logger().get_failed_count()
    if failed_count > 0:
        _print(f"\n  [WARNING] {failed_count} 个任务失败，详见 gen/failed_tasks.json")

    # ══ PHASE 2 ══════════════════════════════

    if args.skip_standardization:
        _print("\n[Step 5] 跳过 OMOP 标准化 (--skip-standardization)")
        std_result = {}
        timer.lap("Step 5 - skipped")
    else:
        if args.skip_omop_matching:
            _print("\n[Step 5] 快速标准化：跳过 OMOP matching，复用已有映射表...")
        else:
            _print("\n[Step 5] OMOP 标准化 (enable_review=True)...")
        std_result = run_standardization(
            gen_dir=gen_dir,
            enable_review=not args.skip_omop_matching,
            skip_omop_matching=args.skip_omop_matching,
        )
        timer.lap("Step 5 - OMOP 标准化")

        _print(f"  原始术语数:    {std_result.get('original_term_count', '?')}")
        _print(f"  标准化后术语数: {std_result.get('final_term_count', '?')}")
        _print(f"  同义术语组:    {std_result.get('synonym_groups', '?')}")
        _print(f"  跳过 OMOP 匹配: {std_result.get('omop_matching_skipped', False)}")

    # ══ 时间汇总 ═════════════════════════════

    timing = timer.summary()
    timing["timestamp"] = timestamp
    timing["gen_dir"] = gen_dir
    timing["input_texts"] = len(texts)
    timing["provenances"] = len(provenances)
    timing["clusters"] = len(clusters)
    timing["failed_tasks"] = failed_count
    timing["phase1"] = {
        "terms": len(extraction_result.get("terms", [])),
        "med_terms": len(extraction_result.get("med_terms", [])),
        "predicates": len(extraction_result.get("predicates", [])),
        "rules": len(extraction_result.get("rules", [])),
    }
    timing["phase2"] = std_result
    timing["excel_path"] = str(excel_path)

    timing_path = os.path.join(gen_dir, "run_timing.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)

    _print("\n" + "=" * 65)
    _print("测试完成！用时汇总:")
    for rec in timing["steps"]:
        _print(f"  {rec['step']:<40} {rec['seconds']:>7.1f}s")
    _print(f"  {'总计':<40} {timing['total_seconds']:>7.1f}s")
    _print(f"\n结果目录: {gen_dir}")
    _print(f"用时记录: {timing_path}")
    _print("=" * 65)

    if args.run_eval:
        try:
            _run_source_grounded_eval(
                gen_dir,
                dry_run=args.eval_dry_run,
                max_items=args.eval_max_items,
            )
        except Exception as exc:
            _print(f"\n[Eval] 失败: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
