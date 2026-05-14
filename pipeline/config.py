"""
配置管理模块 - 集中管理所有配置项
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.05  # 略低以提高结构化抽取（术语/谓词/规则）稳定性
    max_tokens: int = 16384  # 增加输出长度限制
    timeout: float = 300.0  # 5分钟超时，DashScope 响应可能较慢
    max_retries: int = 3

    @classmethod
    def from_env(cls, *, temperature: Optional[float] = None) -> "LLMConfig":
        """从环境变量构建；未设置的字段使用本类默认值。

        环境变量：LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE（可选）。
        若传入 ``temperature``，则强制使用该值（评估脚本等场景）。
        """
        defaults = cls()
        resolved_temp = defaults.temperature
        if temperature is not None:
            resolved_temp = temperature
        else:
            env_temp = os.getenv("LLM_TEMPERATURE")
            if env_temp:
                try:
                    resolved_temp = float(env_temp)
                except ValueError:
                    pass
        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", defaults.base_url),
            model=os.getenv("LLM_MODEL", defaults.model),
            temperature=resolved_temp,
            max_tokens=defaults.max_tokens,
            timeout=defaults.timeout,
            max_retries=defaults.max_retries,
        )


@dataclass
class PathConfig:
    """路径配置"""
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    standard_dir: str = field(default="")
    gen_dir: str = field(default="")
    
    def __post_init__(self):
        if not self.standard_dir:
            self.standard_dir = os.path.join(self.base_dir, "standard")
        if not self.gen_dir:
            self.gen_dir = os.path.join(self.base_dir, "gen")


@dataclass
class MatchConfig:
    """匹配配置

    OMOP 匹配分数通常为 0–100。CombinedTermProcessor 中：
    - 分数 < term_threshold / med_threshold：视为无匹配，不注册 OMOP。
    - 分数 >= high_confidence_skip_review：信任匹配器，不调用 LLM 审核。
    - 介于两者之间：在 enable_review=True 时调用 LLM 审核。

    提高映射「准度」、减少误匹配：略提高 term/med_threshold。
    减少 LLM 审核次数：略降低 high_confidence_skip_review（更多分数区间直接采纳）。
    review_threshold 保留供未来扩展；当前主逻辑以 high_confidence_skip_review 为准。
    """
    term_threshold: float = 85.0
    med_threshold: float = 85.0
    predicate_fuzzy_threshold: float = 70.0
    predicate_high_confidence: float = 90.0
    # LLM 审核（仅 CombinedTermProcessor 标准化路径）
    enable_review: bool = True
    review_threshold: float = 95.0  # 预留：与 UI/报表说明用
    # 匹配分 >= 此值则跳过 LLM 审核（默认 92：比原 98 更易达到，显著减少审核调用）
    high_confidence_skip_review: float = 92.0


@dataclass
class PipelineConfig:
    """流水线总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """从环境变量创建配置（LLM 部分见 LLMConfig.from_env）。"""
        return cls(llm=LLMConfig.from_env())


# 默认配置单例
_default_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """获取默认配置（首次从环境变量加载 LLM 段，与显式 set_config 并存）。"""
    global _default_config
    if _default_config is None:
        _default_config = PipelineConfig.from_env()
    return _default_config


def set_config(config: PipelineConfig):
    """设置默认配置"""
    global _default_config
    _default_config = config

