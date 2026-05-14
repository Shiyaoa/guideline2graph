# -*- coding: utf-8 -*-
"""
从 RAG refined_enriched JSONL 读取 chunk，逐条调用 extract_provenances_stage（每 chunk 一次 LangGraph），
便于将推荐意见与 chunk_id 对齐。

传给 LLM 的用户正文**仅包含** chunk 的 `title` 与 `content`（Markdown 标题 + 正文），不包含 id、filename 等元数据。

用法（在 guideline2graph 目录下）:
  set PYTHONPATH=...\\guideline2graph
  python scripts/manual_tests/extract_jsonl_provenances.py --max-per-file 0 --workers 5 \\
    --out-dir gen/jsonl_provenance_full \\
    --jsonl ..\\..\\RAG\\chunks\\refined_enriched\\eln2022_aml_sections.jsonl

环境变量：LLM_API_KEY 或 DASHSCOPE_API_KEY；可选 LLM_BASE_URL、LLM_MODEL。
会尝试加载 guideline2graph/.env。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_G2G_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_G2G_ROOT)
sys.path.insert(0, str(_G2G_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_G2G_ROOT / ".env")
except ImportError:
    pass

from pipeline.config import LLMConfig, PathConfig, PipelineConfig, set_config
from pipeline.graph_api import extract_provenances_stage


def _resolve_api_key() -> str:
    return (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()


def _build_chunk_text(row: dict) -> str:
    """仅 title + content，不传 id / filename 等，供 RECOMMENDATION_PROMPT 做原文摘录。"""
    title = row.get("title") or ""
    content = row.get("content") or ""
    if title:
        return f"# {title}\n\n{content}".strip()
    return content.strip()


def load_chunk_rows(path: Path, max_per_file: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_per_file > 0 and len(rows) >= max_per_file:
                break
            rows.append(json.loads(line))
    return rows


def setup_llm(gen_dir: Path) -> None:
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError("需要 LLM_API_KEY 或 DASHSCOPE_API_KEY（可在 guideline2graph/.env 中配置）")
    gen_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(
        llm=LLMConfig(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip(),
            model=os.getenv("LLM_MODEL", "deepseek-v4-pro").strip(),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
            timeout=float(os.getenv("LLM_TIMEOUT", "300")),
        ),
        paths=PathConfig(gen_dir=str(gen_dir)),
    )
    set_config(cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="JSONL chunk → extract_provenances_stage（逐 chunk）")
    p.add_argument(
        "--jsonl",
        nargs="+",
        type=Path,
        required=True,
        help="一个或多个 .jsonl 路径",
    )
    p.add_argument(
        "--max-per-file",
        type=int,
        default=3,
        help="每个文件最多处理多少行；0 表示不限制（全量）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="同一文件内并发抽取的线程数（每线程独立 invoke）；全量时可酌情调高并注意 API 限流",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录；默认 gen/jsonl_provenance_<timestamp>",
    )
    p.add_argument("--max-concurrency", type=int, default=3)
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or (_G2G_ROOT / "gen" / f"jsonl_provenance_{ts}")).resolve()
    setup_llm(out_dir)

    summary: dict = {"out_dir": str(out_dir), "files": []}

    for jp in args.jsonl:
        jp = jp.resolve()
        rows = load_chunk_rows(jp, args.max_per_file)
        indexed = [(i, row) for i, row in enumerate(rows)]

        def _one(item: tuple[int, dict]) -> tuple[int, dict]:
            i, row = item
            cid = str(row.get("id", ""))
            text = _build_chunk_text(row)
            provs = extract_provenances_stage(
                [text],
                save_to_file=False,
                max_concurrency=args.max_concurrency,
            )
            rec = {
                "chunk_id": cid,
                "title": row.get("title"),
                "filename": row.get("filename"),
                "provenance_count": len(provs),
                "provenances": [pr.model_dump(mode="json") for pr in provs],
            }
            print(f"  [{jp.name}] {cid} → {len(provs)} provenance(s)", flush=True)
            return i, rec

        file_recs: list[dict | None] = [None] * len(indexed)
        if args.workers <= 1:
            for item in indexed:
                i, rec = _one(item)
                file_recs[i] = rec
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_one, item): item[0] for item in indexed}
                for fut in as_completed(futures):
                    i, rec = fut.result()
                    file_recs[i] = rec
        file_recs = [r for r in file_recs if r is not None]

        out_path = out_dir / f"{jp.stem}_provenances_by_chunk.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(file_recs, f, ensure_ascii=False, indent=2)

        total_p = sum(r["provenance_count"] for r in file_recs)
        summary["files"].append(
            {
                "jsonl": str(jp),
                "chunks": len(file_recs),
                "total_provenances": total_p,
                "output": str(out_path),
            }
        )
        print(f"[{jp.name}] wrote {out_path} ({len(file_recs)} chunks, {total_p} provenances)\n")

    meta_path = out_dir / "summary.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary: {meta_path}")


if __name__ == "__main__":
    main()
