"""
标准库管理器 - 启动时一次性加载所有标准库到内存
"""
import os
import json
import threading
from typing import Dict, List, Optional

from .config import get_config, PipelineConfig
from .library_registry import (
    get_registered_library_function_ids,
    get_v2_library_function_specs,
    is_registered_library_function,
)
from .models import LibraryFunction


V2_LIBRARY_FUNCTIONS: List[LibraryFunction] = [
    LibraryFunction(**spec) for spec in get_v2_library_function_specs()
]



class StandardLibrary:
    """
    标准库内存管理器 - 单例模式
    启动时一次性加载所有标准库到内存，避免重复IO操作

    线程安全的单例实现
    """
    _instance: Optional["StandardLibrary"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[PipelineConfig] = None):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        if self._initialized:
            return
        
        self._config = config or get_config()
        self._standard_dir = self._config.paths.standard_dir
        
        # 一次性加载所有标准库到内存
        self.terms: List[dict] = self._load_json("terms.json")
        self.meds: List[dict] = self._load_json("meds.json")
        self.predicates: List[dict] = self._load_json("predicates.json")
        self.library_functions: List[LibraryFunction] = V2_LIBRARY_FUNCTIONS.copy()
        
        # 构建索引（用于快速查找）
        self._build_indices()
        
        # Z3环境（懒加载）
        
        self._initialized = True
        print(f"[StandardLibrary] 已加载: {len(self.terms)} terms, "
              f"{len(self.meds)} meds, {len(self.predicates)} predicates, "
              f"{len(self.library_functions)} v2 library functions")
    
    def _load_json(self, filename: str) -> List[dict]:
        """加载JSON文件"""
        filepath = os.path.join(self._standard_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[StandardLibrary] 警告: 文件不存在 {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"[StandardLibrary] 警告: JSON解析错误 {filepath}: {e}")
            return []
    
    def _build_indices(self):
        """构建索引以支持快速查找"""
        # Terms 索引
        self.terms_by_id: Dict[str, dict] = {t['id']: t for t in self.terms}
        self.terms_by_name: Dict[str, dict] = {t['name'].lower(): t for t in self.terms}
        
        # 添加别名索引
        for t in self.terms:
            for alias in t.get('aliases', []):
                self.terms_by_name[alias.lower()] = t
        
        # Meds 索引
        self.meds_by_id: Dict[str, dict] = {m['id']: m for m in self.meds}
        self.meds_by_name: Dict[str, dict] = {m['name'].lower(): m for m in self.meds}
        
        # Predicates 索引
        self.predicates_by_id: Dict[str, dict] = {p['id']: p for p in self.predicates}
        self.predicates_by_label: Dict[str, dict] = {
            p.get('label', '').lower(): p for p in self.predicates if p.get('label')
        }

        self.library_functions_by_id: Dict[str, LibraryFunction] = {
            fn.id: fn for fn in self.library_functions
        }
    
    def get_term_names(self) -> List[str]:
        """获取所有术语名称（小写）"""
        return list(self.terms_by_name.keys())
    
    def get_med_names(self) -> List[str]:
        """获取所有药物名称（小写）"""
        return list(self.meds_by_name.keys())
    
    def get_predicate_labels(self) -> List[str]:
        """获取所有谓词标签（小写）"""
        return list(self.predicates_by_label.keys())

    def get_library_functions(self) -> List[LibraryFunction]:
        """获取 v2 标准 library function 清单。"""
        return list(self.library_functions)

    def get_library_function(self, function_id: str) -> Optional[LibraryFunction]:
        """按 ID 获取 v2 标准 library function。"""
        return self.library_functions_by_id.get(function_id)
    
    @classmethod
    def reset(cls):
        """重置单例（主要用于测试）"""
        cls._instance = None


# 便捷函数
def get_standard_library(config: Optional[PipelineConfig] = None) -> StandardLibrary:
    """获取标准库单例"""
    return StandardLibrary(config)


def get_v2_library_functions() -> List[LibraryFunction]:
    """返回 v2 最小标准 library function 清单。"""
    return list(V2_LIBRARY_FUNCTIONS)


def get_v2_library_function_ids() -> List[str]:
    """返回已注册的 v2 library function id 清单。"""
    return get_registered_library_function_ids()


def is_v2_library_function_registered(function_id: Optional[str]) -> bool:
    """检查 library function id 是否已注册。"""
    return is_registered_library_function(function_id)

