#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMOP中文术语标准化工具包 - 使用示例

演示如何使用omop_normalizer包进行术语标准化
"""

import os
import sys

# 添加包路径（开发模式下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omop_normalizer import ChineseTermNormalizer


def demo_basic():
    """基础用法示例"""
    print("=" * 60)
    print("示例1: 基础用法")
    print("=" * 60)

    # 初始化标准化器
    normalizer = ChineseTermNormalizer(
        concept_csv_path="CONCEPT.csv",
        cache_db_path="demo_cache.db",
        llm_api_type="qwen",
        confidence_threshold=0.75,
        verbose=True
    )

    # 测试文本
    text = """
    2型糖尿病患者推荐使用SGLT2抑制剂。
    首选二甲双胍口服治疗，血糖控制不佳时可加用胰岛素。
    心衰患者应使用β受体阻滞剂，并监测血压变化。
    """

    # 标准化
    results = normalizer.normalize(text)

    # 输出结果
    print("\n" + "=" * 60)
    print("标准化结果")
    print("=" * 60)
    for r in results:
        print(f"\n中文: {r.chinese_term}")
        print(f"英文: {r.english_term}")
        print(f"概念ID: {r.concept_id}")
        print(f"标准名: {r.concept_name}")
        print(f"领域: {r.domain_id}")
        print(f"状态: {r.status}")
        print(f"置信度: {r.final_confidence:.2f}")


def demo_batch():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("示例2: 批量处理")
    print("=" * 60)

    normalizer = ChineseTermNormalizer(
        concept_csv_path="CONCEPT.csv",
        cache_db_path="demo_cache.db",
        llm_api_type="qwen",
        verbose=True
    )

    # 批量文本
    texts = [
        ("推荐使用ACEI或ARB治疗高血压伴糖尿病患者", "高血压指南"),
        ("心衰住院患者应评估心功能分级", "心衰指南"),
        ("CKD患者需定期监测eGFR和蛋白尿", "CKD指南"),
    ]

    # 批量处理
    results = normalizer.batch_normalize(texts)

    # 统计
    print("\n" + "=" * 60)
    print("处理统计")
    print("=" * 60)
    stats = normalizer.get_statistics()
    print(f"总映射数: {stats['total_mappings']}")
    print(f"已验证数: {stats['verified_mappings']}")
    print(f"待审核数: {stats['pending_reviews']}")
    print(f"按领域分布: {stats['by_domain']}")

    # 导出
    normalizer.export_results(results, "demo_results.xlsx")
    print(f"\n结果已导出到 demo_results.xlsx")


def demo_search():
    """概念搜索示例"""
    print("\n" + "=" * 60)
    print("示例3: 概念搜索")
    print("=" * 60)

    normalizer = ChineseTermNormalizer(
        concept_csv_path="CONCEPT.csv",
        enable_review=False,  # 禁用审核，加快速度
        verbose=False
    )

    # 搜索概念
    keywords = ["diabetes", "metformin", "heart failure", "SGLT2"]

    for kw in keywords:
        print(f"\n搜索: {kw}")
        matches = normalizer.search_concept(kw, top_k=3)
        for m in matches:
            print(f"  - {m.concept_name[:50]} (ID: {m.concept_id}, 分数: {m.match_score:.1f})")


def demo_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 60)
    print("示例4: 自定义配置")
    print("=" * 60)

    # 使用不同的LLM后端
    configs = [
        {"api_type": "qwen", "model": "qwen-max"},
        {"api_type": "deepseek", "model": "deepseek-chat"},
        {"api_type": "openai", "model": "gpt-4"},
    ]

    for config in configs:
        print(f"\n配置: {config}")
        try:
            normalizer = ChineseTermNormalizer(
                concept_csv_path="CONCEPT.csv",
                llm_api_type=config["api_type"],
                llm_model=config["model"],
                enable_review=False,
                verbose=False
            )
            print(f"  初始化成功")
        except Exception as e:
            print(f"  初始化失败: {e}")


if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请设置: export DASHSCOPE_API_KEY='your-api-key'")
        print()

    # 运行示例
    print("\nOMOP中文术语标准化工具包 - 使用示例\n")

    # 选择运行哪个示例
    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
        if demo_name == "basic":
            demo_basic()
        elif demo_name == "batch":
            demo_batch()
        elif demo_name == "search":
            demo_search()
        elif demo_name == "config":
            demo_custom_config()
        else:
            print(f"未知示例: {demo_name}")
            print("可用示例: basic, batch, search, config")
    else:
        # 默认运行概念搜索示例（不需要API Key）
        demo_search()