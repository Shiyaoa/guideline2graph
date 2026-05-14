"""
集成测试 - 使用真实 LLM API（通过 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 配置）
从 LSH 聚类开始测试各个 Stage 和完整 Pipeline

数据来源: SGLT2i_zh.xlsx 的指南和专家共识 sheet
"""
import os
import sys
import json
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.models import Provenance, ProvenanceCluster
from pipeline.config import set_config, PipelineConfig, LLMConfig, PathConfig, MatchConfig


# ============ 配置 ============

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")


def get_llm_api_key(required: bool = False) -> str:
    """Read the real API key from the environment."""
    api_key = os.getenv("LLM_API_KEY", "")
    if required and not api_key:
        pytest.skip("LLM_API_KEY is required for real API integration tests")
    return api_key


def get_test_config(temp_dir: str) -> PipelineConfig:
    """获取测试配置"""
    return PipelineConfig(
        llm=LLMConfig(
            api_key=get_llm_api_key(required=False),
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=16384,
            timeout=300.0
        ),
        paths=PathConfig(gen_dir=temp_dir),
        match=MatchConfig(term_threshold=80.0, med_threshold=80.0)
    )


def load_provenances_from_excel(
    excel_path: str,
    max_per_sheet: int = None
) -> list[Provenance]:
    """
    从 Excel 加载推荐意见

    Args:
        excel_path: Excel 文件路径
        max_per_sheet: 每个 sheet 最大加载数量（用于快速测试）

    Returns:
        Provenance 列表
    """
    xl = pd.ExcelFile(excel_path)
    provenances = []

    # Sheet 1 (index 1): 指南
    df_guideline = pd.read_excel(xl, sheet_name=1)
    source_col = df_guideline.columns[0]  # 来源
    quote_col = df_guideline.columns[1]   # 推荐意见

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

    # Sheet 2 (index 2): 专家共识
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

    print(f"[load_provenances] 加载了 {len(provenances)} 条推荐意见")
    return provenances


# ============ Fixtures ============

@pytest.fixture(scope="module")
def excel_path():
    """Excel 文件路径"""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "SGLT2i_zh.xlsx"
    )
    assert os.path.exists(path), f"Excel 文件不存在: {path}"
    return path


@pytest.fixture(scope="module")
def sample_provenances(excel_path):
    """样本推荐意见（用于快速测试）- 每个sheet取5条，共10条"""
    return load_provenances_from_excel(excel_path, max_per_sheet=5)


@pytest.fixture(scope="module")
def all_provenances(excel_path):
    """全部推荐意见"""
    return load_provenances_from_excel(excel_path)


@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)


@pytest.fixture(autouse=True)
def setup_config(temp_output_dir):
    """设置测试配置"""
    config = get_test_config(temp_output_dir)
    set_config(config)
    yield config
