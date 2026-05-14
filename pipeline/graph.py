"""
临床指南知识抽取工作流 - 支持多轮输入 + LSH 聚类 + 并行子图

实现拆分为 ``llm_factory``、``structured_llm``、``graph_prompts``、
``graph_nodes``、``graph_builders``；本模块继续 re-export 原 ``pipeline.graph``
公开符号以保持兼容。
"""
from .graph_builders import (
    ASYNC_CLUSTER_SUBGRAPH,
    CLUSTER_SUBGRAPH,
    async_build_extraction_subgraph,
    aprocess_cluster,
    build_extraction_subgraph,
    build_pipeline_graph,
    build_predicates_extraction_graph,
    build_rules_extraction_graph,
    build_terms_extraction_graph,
    distribute_clusters_for_predicates,
    distribute_clusters_for_rules,
    distribute_clusters_for_terms,
    process_cluster,
    process_cluster_node,
    process_cluster_predicates,
    process_cluster_rules,
    process_cluster_terms,
    route_to_clusters,
)
from .graph_nodes import (
    asubgraph_extract_all_terms,
    async_extract_predicates_subgraph,
    async_extract_rules_subgraph,
    distribute_texts,
    do_lsh_clustering,
    extract_recommendation,
)
from .llm_factory import _chat_openai_kwargs, _create_llm, _get_default_llm
from .models import merge_by_id, merge_cluster_cache_updates, _to_models
from .structured_llm import (
    _ainvoke_structured_list,
    _ainvoke_structured_output_direct,
    _aretry_term_extraction_repair,
    _context_json,
    _extract_langchain_metadata,
    _extract_raw_output,
    _invoke_structured_list,
    _invoke_structured_output_direct,
    _pydantic_to_function_schema,
    _retry_term_extraction_repair,
    _safe_items,
)

__all__ = [
    "ASYNC_CLUSTER_SUBGRAPH",
    "CLUSTER_SUBGRAPH",
    "_ainvoke_structured_list",
    "_ainvoke_structured_output_direct",
    "_aretry_term_extraction_repair",
    "_chat_openai_kwargs",
    "_context_json",
    "_create_llm",
    "_extract_langchain_metadata",
    "_extract_raw_output",
    "_get_default_llm",
    "_invoke_structured_list",
    "_invoke_structured_output_direct",
    "_pydantic_to_function_schema",
    "_retry_term_extraction_repair",
    "_safe_items",
    "_to_models",
    "aprocess_cluster",
    "asubgraph_extract_all_terms",
    "async_build_extraction_subgraph",
    "async_extract_predicates_subgraph",
    "async_extract_rules_subgraph",
    "build_extraction_subgraph",
    "build_pipeline_graph",
    "build_predicates_extraction_graph",
    "build_rules_extraction_graph",
    "build_terms_extraction_graph",
    "distribute_clusters_for_predicates",
    "distribute_clusters_for_rules",
    "distribute_clusters_for_terms",
    "distribute_texts",
    "do_lsh_clustering",
    "extract_recommendation",
    "merge_by_id",
    "merge_cluster_cache_updates",
    "process_cluster",
    "process_cluster_node",
    "process_cluster_predicates",
    "process_cluster_rules",
    "process_cluster_terms",
    "route_to_clusters",
]
