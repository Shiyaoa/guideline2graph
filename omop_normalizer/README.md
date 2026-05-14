# OMOP Chinese Term Normalizer

中文医学术语标准化工具包，用于将中文临床指南文本映射到 OMOP CDM 标准词汇表。

## 安装

```bash
# 基础安装
pip install -e .

# 安装所有可选依赖
pip install -e ".[all]"
```

## 快速开始

```python
from omop_normalizer import ChineseTermNormalizer

# 初始化标准化器
normalizer = ChineseTermNormalizer(
    concept_csv_path="CONCEPT.csv",      # OMOP概念表路径
    cache_db_path="term_cache.db",       # 缓存数据库路径
    llm_api_type="qwen",                 # LLM类型: qwen, deepseek, openai, anthropic
    confidence_threshold=0.75            # 置信度阈值
)

# 标准化单段文本
text = """
2型糖尿病患者推荐使用SGLT2抑制剂。
首选二甲双胍治疗，血糖控制不佳时可加用胰岛素。
"""

results = normalizer.normalize(text)

# 查看结果
for r in results:
    print(f"中文: {r.chinese_term}")
    print(f"英文: {r.english_term}")
    print(f"概念ID: {r.concept_id}")
    print(f"标准名: {r.concept_name}")
    print(f"领域: {r.domain_id}")
    print(f"状态: {r.status}")
    print(f"置信度: {r.final_confidence:.2f}")
    print("-" * 40)
```

## 批量处理

```python
# 批量处理多段文本
texts = [
    ("推荐使用ACEI或ARB治疗高血压", "指南A"),
    ("心衰患者应使用β受体阻滞剂", "指南B"),
    ("CKD患者需监测eGFR水平", "指南C"),
]

results = normalizer.batch_normalize(texts)

# 导出结果到Excel
normalizer.export_results(results, "normalization_results.xlsx")
```

## 核心流程

```
中文文本 → LLM术语提取 → OMOP概念匹配 → LLM审核 → 缓存存储
```

### Step 1: LLM术语提取

从中文文本中识别医学术语，并给出标准英文名。

### Step 2: OMOP概念匹配

使用多种策略匹配OMOP标准概念：

| 策略 | 说明 | 分数范围 |
|------|------|----------|
| 特殊映射 | 预定义的医学缩写映射 | 98 |
| 缩写扩展 | 自动扩展缩写 | 95-100 |
| 中文词典 | 中文→英文映射 | 85-95 |
| 精确匹配 | 英文名完全匹配 | 100 |
| 包含匹配 | 名称互相包含 | 80-90 |
| 模糊匹配 | 相似度计算 | ≥70 |
| 药物类别 | 类别→代表药物 | 80 |

### Step 3: LLM审核

验证匹配结果是否正确，计算最终置信度。

### Step 4: 缓存存储

审核通过的映射存入SQLite缓存，避免重复处理。

## 扩展词典

可在 `dictionaries.py` 中扩展词典：

```python
# 添加中文映射
CHINESE_ENGLISH_DICT["新术语"] = "new term"

# 添加缩写
ABBREVIATION_DICT["新缩写"] = "Full Name"

# 添加特殊概念映射
SPECIAL_MAPPINGS[("关键词", "Drug")] = 12345  # concept_id
```

## API配置

支持多种LLM后端，通过环境变量配置：

```bash
# 阿里云通义千问
export DASHSCOPE_API_KEY="your-api-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-api-key"

# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"
```

## 输出字段说明

| 字段 | 说明 |
|------|------|
| chinese_term | 中文术语原文 |
| english_term | LLM给出的英文术语 |
| concept_id | OMOP概念ID |
| concept_name | OMOP标准概念名 |
| domain_id | 领域（Condition/Drug/Procedure等） |
| vocabulary_id | 来源词汇表（SNOMED/RxNorm等） |
| extraction_confidence | 提取置信度 |
| match_score | 匹配分数 |
| review_confidence | 审核置信度 |
| final_confidence | 最终综合置信度 |
| status | 状态（approved/rejected/needs_review/cached/no_match） |
| match_type | 匹配类型 |

## 项目结构

```
omop_normalizer/
├── __init__.py         # 包入口
├── models.py           # 数据模型
├── dictionaries.py     # 医学词典配置
├── matcher.py          # OMOP概念匹配器
├── extractor.py        # LLM术语提取器
├── cache.py            # 映射缓存管理
├── normalizer.py       # 主标准化器
├── setup.py            # 安装配置
├── requirements.txt    # 依赖列表
└── README.md           # 说明文档
```

## 依赖

- pandas >= 1.3.0
- requests >= 2.25.0
- openai >= 1.0.0
- rapidfuzz >= 2.0.0 (可选，用于模糊匹配)

## OMOP 词汇表数据文件

本仓库不包含 `omop_normalizer/` 下的 Athena 导出 CSV（`CONCEPT.csv`、`CONCEPT_RELATIONSHIP.csv` 等），因体积超过 GitHub 限制。请从 [OHDSI Athena](https://athena.ohdsi.org/) 下载所需词汇表，将对应 CSV 放在 `omop_normalizer/` 目录后再运行匹配与标准化流程。运行过程中可能生成 `*.pkl` 缓存文件，已加入 `.gitignore`，勿提交到版本库。

## 许可证

MIT License