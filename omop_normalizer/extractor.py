"""
LLM术语提取器

支持多种LLM后端：
- 阿里云通义千问 (Qwen)
- DeepSeek
- OpenAI
- Anthropic Claude
"""

import json
import re
import os
from typing import List, Optional
from dataclasses import dataclass

from .models import ExtractedTerm, ReviewResult


class LLMTermExtractor:
    """LLM术语提取器"""

    EXTRACTION_PROMPT = """你是一个医学信息学专家，擅长从临床指南文本中识别和标准化医学术语。

## 任务
从以下中文临床指南文本中提取医学术语，并给出对应的标准英文名称。

## 关注的实体类型
1. **Condition（诊断/疾病）**: 疾病名称、综合征、症状、体征
2. **Drug（药物）**: 药品名称、活性成分、药物剂型
3. **Procedure（医疗程序）**: 手术、操作、检查、治疗程序
4. **Measurement（测量）**: 实验室指标、测量值、定量评估
5. **Observation（观察）**: 临床观察、定性评估、患者状态

## 输出要求
严格按照JSON格式输出，不要添加任何其他内容：
{"entities": [{"chinese_term": "中文术语", "entity_type": "类型", "english_standard": "标准英文名", "confidence": 0.95}]}

## 输入文本
{text}
"""

    REVIEW_PROMPT = """你是一个医学术语标准化专家，请审核以下术语映射是否正确。

## 中文术语
{chinese_term}

## LLM给出的英文术语
{english_term}

## OMOP匹配结果
概念名称: {concept_name}
领域: {domain}

## 审核要求
1. 判断OMOP概念名称是否与中文术语含义一致
2. 考虑医学术语的专业性和准确性

## 输出格式（严格JSON）
{{"is_correct": true/false, "correct_concept_name": "正确概念名(如不匹配)", "reason": "判断理由", "confidence": 0.9}}
"""

    def __init__(self, api_type: str = "qwen", api_key: Optional[str] = None,
                 model: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化LLM提取器

        Args:
            api_type: API类型 (qwen, deepseek, openai, anthropic)
            api_key: API密钥
            model: 模型名称
            base_url: 自定义API地址
        """
        self.api_type = api_type
        self.api_key = api_key or self._get_default_key(api_type)
        self.model = model or self._get_default_model(api_type)
        self.base_url = base_url or self._get_default_url(api_type)
        self._client = None

    def _get_default_key(self, api_type: str) -> str:
        """获取默认API Key"""
        key_map = {
            "qwen": "DASHSCOPE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        return os.getenv(key_map.get(api_type, "LLM_API_KEY"), "")

    def _get_default_model(self, api_type: str) -> str:
        """获取默认模型"""
        model_map = {
            "qwen": "qwen-max",
            "deepseek": "deepseek-chat",
            "openai": "gpt-4",
            "anthropic": "claude-3-sonnet-20240229",
        }
        return model_map.get(api_type, "gpt-4")

    def _get_default_url(self, api_type: str) -> str:
        """获取默认API地址"""
        url_map = {
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "openai": "https://api.openai.com/v1",
        }
        return url_map.get(api_type, "")

    def _init_client(self):
        """延迟初始化客户端"""
        if self._client is not None:
            return

        if self.api_type == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        else:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def _call_api(self, prompt: str, temperature: float = 0.1) -> str:
        """调用API"""
        self._init_client()

        if self.api_type == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content

    def extract_terms(self, text: str) -> List[ExtractedTerm]:
        """
        从文本中提取医学术语

        Args:
            text: 中文临床指南文本

        Returns:
            提取的术语列表
        """
        try:
            prompt = self.EXTRACTION_PROMPT.format(text=text)
            response = self._call_api(prompt, temperature=0.1)

            # 解析JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                entities = data.get("entities", [])

                return [ExtractedTerm(
                    chinese_text=e.get("chinese_term", "").strip(),
                    entity_type=e.get("entity_type", "Condition"),
                    english_standard=e.get("english_standard", "").strip(),
                    confidence=float(e.get("confidence", 0.8)),
                    source_text=text[:200]
                ) for e in entities if e.get("chinese_term")]
        except Exception as e:
            print(f"术语提取失败: {e}")

        return []

    def review_match(self, chinese_term: str, english_term: str,
                     concept_name: str, domain: str,
                     context: str = "") -> ReviewResult:
        """
        审核OMOP匹配结果

        Args:
            chinese_term: 中文术语
            english_term: 英文术语
            concept_name: OMOP概念名称
            domain: 领域
            context: 上下文

        Returns:
            审核结果
        """
        try:
            prompt = self.REVIEW_PROMPT.format(
                chinese_term=chinese_term,
                english_term=english_term,
                concept_name=concept_name,
                domain=domain
            )
            response = self._call_api(prompt, temperature=0)

            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return ReviewResult(
                    is_correct=data.get("is_correct", False),
                    correct_concept_name=data.get("correct_concept_name", ""),
                    reason=data.get("reason", ""),
                    confidence=float(data.get("confidence", 0.5))
                )
            return ReviewResult(
                is_correct=False,
                correct_concept_name="",
                reason="LLM review response did not contain valid JSON.",
                confidence=0
            )
        except Exception as e:
            print(f"审核失败: {e}")
            return ReviewResult(
                is_correct=False,
                correct_concept_name="",
                reason=str(e),
                confidence=0
            )

    def disambiguate(self, chinese_term: str, english_term: str,
                     candidates: List[dict], context: str = "") -> Optional[dict]:
        """
        使用LLM进行概念消歧

        Args:
            chinese_term: 中文术语
            english_term: 英文术语
            candidates: 候选概念列表
            context: 上下文

        Returns:
            最佳匹配概念
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        candidates_str = "\n".join([
            f"{i+1}. {c['concept_name']} (ID: {c['concept_id']}, Domain: {c['domain_id']})"
            for i, c in enumerate(candidates[:10])
        ])

        prompt = f"""你是一个医学术语标准化专家。请从候选概念中选择最佳匹配。

中文术语: {chinese_term}
英文术语: {english_term}
上下文: {context}

候选概念:
{candidates_str}

请仅返回最匹配的候选编号(1-{len(candidates)})，不要返回其他内容。"""

        try:
            response = self._call_api(prompt, temperature=0)
            choice = int(response.strip())
            if 1 <= choice <= len(candidates):
                return candidates[choice - 1]
        except Exception as e:
            print(f"消歧失败: {e}")

        return candidates[0]