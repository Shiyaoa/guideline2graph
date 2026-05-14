#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMOP中文术语标准化工具 - 命令行接口
"""

import argparse
import sys
import os

from .normalizer import ChineseTermNormalizer


def main():
    parser = argparse.ArgumentParser(
        description="OMOP中文医学术语标准化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 标准化单个文本
  omop-normalize -t "2型糖尿病患者推荐使用SGLT2抑制剂"

  # 批量处理Excel文件
  omop-normalize -f input.xlsx -o results.xlsx

  # 搜索概念
  omop-normalize --search "diabetes" --domain Condition
        """
    )

    parser.add_argument("-t", "--text", type=str, help="要标准化的中文文本")
    parser.add_argument("-f", "--file", type=str, help="输入Excel文件路径")
    parser.add_argument("-o", "--output", type=str, default="results.xlsx", help="输出文件路径")
    parser.add_argument("-c", "--concept-csv", type=str, default="CONCEPT.csv", help="OMOP CONCEPT.csv路径")
    parser.add_argument("--cache-db", type=str, default="term_cache.db", help="缓存数据库路径")
    parser.add_argument("--api-type", type=str, default="qwen", choices=["qwen", "deepseek", "openai", "anthropic"], help="LLM API类型")
    parser.add_argument("--model", type=str, help="LLM模型名称")
    parser.add_argument("--threshold", type=float, default=0.75, help="置信度阈值")
    parser.add_argument("--no-review", action="store_true", help="禁用LLM审核")
    parser.add_argument("--search", type=str, help="搜索模式：搜索关键词")
    parser.add_argument("--domain", type=str, help="领域过滤 (Condition/Drug/Procedure/Measurement/Observation)")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    # 初始化标准化器
    try:
        normalizer = ChineseTermNormalizer(
            concept_csv_path=args.concept_csv,
            cache_db_path=args.cache_db,
            llm_api_type=args.api_type,
            llm_model=args.model,
            confidence_threshold=args.threshold,
            enable_review=not args.no_review,
            verbose=args.verbose
        )
    except FileNotFoundError:
        print(f"错误: 找不到CONCEPT.csv文件: {args.concept_csv}")
        sys.exit(1)

    # 搜索模式
    if args.search:
        matches = normalizer.search_concept(args.search, args.domain, args.top_k)
        print(f"\n搜索结果: {args.search}")
        print("-" * 60)
        for m in matches:
            print(f"ID: {m.concept_id}")
            print(f"名称: {m.concept_name}")
            print(f"领域: {m.domain_id}")
            print(f"词汇表: {m.vocabulary_id}")
            print(f"分数: {m.match_score:.1f}")
            print("-" * 60)
        return

    # 单文本模式
    if args.text:
        results = normalizer.normalize(args.text)
        normalizer.export_results(results, args.output)
        print(f"\n结果已保存到: {args.output}")
        return

    # 批量文件模式
    if args.file:
        import pandas as pd
        df = pd.read_excel(args.file)
        text_col = df.columns[0] if len(df.columns) > 0 else "text"

        texts = [(str(row[text_col]), f"row_{i}") for i, row in df.iterrows()]
        results = normalizer.batch_normalize(texts)
        normalizer.export_results(results, args.output)
        print(f"\n处理完成，结果已保存到: {args.output}")
        return

    # 无参数则显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()