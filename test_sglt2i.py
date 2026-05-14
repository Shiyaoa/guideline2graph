# -*- coding: utf-8 -*-
"""
测试脚本：使用 SGLT2i_zh.xlsx 指南 sheet 的前 10 条数据进行 pipeline 测试
每次传递 5 条数据
"""
import os
import sys
import pandas as pd
from datetime import datetime

# 确保正确的工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置 API Key（需要用户提供）
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

if not API_KEY:
    print("请设置环境变量 DASHSCOPE_API_KEY 或在脚本中直接填写 API Key")
    sys.exit(1)

def main():
    print("=" * 60)
    print("SGLT2i 指南 Pipeline 测试")
    print("=" * 60)

    # 读取数据
    print("\n[1] 读取 Excel 数据...")
    df = pd.read_excel('SGLT2i_zh.xlsx', sheet_name=1)
    print(f"    总共 {len(df)} 条推荐意见")

    # 取前 10 条
    df_test = df.head(10)
    texts = df_test['推荐意见'].tolist()

    print(f"    测试前 {len(texts)} 条数据")
    print("-" * 60)

    # 导入 pipeline
    from pipeline import create_pipeline, save_to_gen, get_failed_task_logger

    # 清除旧的失败日志
    get_failed_task_logger().clear()
    print("\n[!] 已清除旧的失败日志")

    # 创建 pipeline
    print("\n[2] 创建 Pipeline (qwen3-max)...")
    pipeline = create_pipeline(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3-max"
    )
    print("    Pipeline 创建成功")

    # 分两批处理，每批 5 条
    batch_size = 5
    all_results = []

    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        print(f"\n[3.{batch_num}] 处理第 {batch_num}/{total_batches} 批数据 ({len(batch_texts)} 条)...")
        print("-" * 40)

        for i, t in enumerate(batch_texts):
            preview = t[:60] + "..." if len(t) > 60 else t
            print(f"    [{batch_idx + i + 1}] {preview}")

        print()

        # 运行 pipeline
        result = pipeline.run(batch_texts, max_concurrency=3)

        # 显示结果统计
        print(f"\n    结果统计:")
        print(f"    - Terms: {len(result.get('terms', []))}")
        print(f"    - Med Terms: {len(result.get('med_terms', []))}")
        print(f"    - Predicates: {len(result.get('predicates', []))}")
        print(f"    - Rules: {len(result.get('rules', []))}")
        print(f"    - Provenances: {len(result.get('provenance_buffer', []))}")

        all_results.append(result)

    # 合并所有结果并保存
    print("\n[4] 保存结果...")

    # 创建合并后的结果
    merged_result = {
        "terms": [],
        "med_terms": [],
        "predicates": [],
        "rules": [],
        "provenance_buffer": []
    }

    for result in all_results:
        for key in merged_result:
            merged_result[key].extend(result.get(key, []))

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"gen/sglt2i_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    from pipeline import save_to_gen
    save_to_gen(merged_result, gen_dir=output_dir)

    print(f"\n[5] 测试完成!")
    print(f"    结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()