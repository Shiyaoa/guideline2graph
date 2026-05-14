"""Source-grounded semantic equivalence evaluation for v2 graph outputs."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import LLMConfig


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class EvalError(EvalModel):
    layer: str = Field(description="recommendation/entity/predicate/rule/action/faithfulness/execution")
    type: str = Field(description="Normalized error type.")
    severity: str = Field(description="minor/major/critical")
    message: str
    source_evidence: Optional[str] = None
    graph_evidence: Optional[str] = None


class LayerScores(EvalModel):
    entity: float = Field(ge=0, le=4)
    predicate: float = Field(ge=0, le=4)
    rule_logic: float = Field(ge=0, le=4)
    action: float = Field(ge=0, le=4)
    faithfulness: float = Field(ge=0, le=4)
    overall: float = Field(ge=0, le=4)
    weighted_total: float = Field(ge=0, le=4)


class DirectionalFaithfulness(EvalModel):
    missing_semantics: List[str] = Field(default_factory=list)
    unsupported_semantics: List[str] = Field(default_factory=list)
    grounded_elements: List[str] = Field(default_factory=list)


class JudgeEvaluation(EvalModel):
    recommendation_id: str
    matched_rule_ids: List[str] = Field(default_factory=list)
    verdict: str = Field(description="complete/minor_errors/major_errors/wrong/unmatched")
    scores: LayerScores
    source_to_graph: DirectionalFaithfulness = Field(default_factory=DirectionalFaithfulness)
    graph_to_source: DirectionalFaithfulness = Field(default_factory=DirectionalFaithfulness)
    predicate_slot_notes: List[str] = Field(default_factory=list)
    action_notes: List[str] = Field(default_factory=list)
    errors: List[EvalError] = Field(default_factory=list)
    rationale: str
    clinical_risk: str = Field(description="low/medium/high")


class GoldRecord(EvalModel):
    recommendation_id: Optional[str] = None
    source_id: Optional[str] = None
    rule_id: Optional[str] = None
    rule_ids: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    source_text: str
    gold_frame: Dict[str, Any]

    @property
    def stable_id(self) -> str:
        return self.recommendation_id or self.source_id or self.rule_id or _stable_text_id(self.source_text)

    @property
    def requested_rule_ids(self) -> List[str]:
        ids = []
        if self.rule_id:
            ids.append(self.rule_id)
        ids.extend(self.rule_ids or [])
        return list(dict.fromkeys(ids))


@dataclass
class PipelineArtifacts:
    gen_dir: Path
    terms: List[Dict[str, Any]]
    med_terms: List[Dict[str, Any]]
    predicates: List[Dict[str, Any]]
    rules: List[Dict[str, Any]]
    provenances: List[Dict[str, Any]]
    cluster_final: Dict[str, Any]

    @classmethod
    def load(cls, gen_dir: str | Path) -> "PipelineArtifacts":
        root = Path(gen_dir)
        return cls(
            gen_dir=root,
            terms=_load_json(root / "terms.json", []),
            med_terms=_load_json(root / "med_terms.json", []),
            predicates=_load_json(root / "predicates.json", []),
            rules=_load_json(root / "rules.json", []),
            provenances=_load_json(root / "provenances.json", []),
            cluster_final=_load_json(root / "cluster_final.json", {}),
        )

    @property
    def predicates_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {item.get("id"): item for item in self.predicates if item.get("id")}

    @property
    def terms_by_id(self) -> Dict[str, Dict[str, Any]]:
        merged = {}
        for item in self.terms + self.med_terms:
            if item.get("id"):
                merged[item["id"]] = item
        return merged

    @property
    def rules_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {item.get("id"): item for item in self.rules if item.get("id")}


@dataclass
class EvaluationContext:
    gold: GoldRecord
    matched_rules: List[Dict[str, Any]]
    matched_predicates: List[Dict[str, Any]]
    matched_terms: List[Dict[str, Any]]
    unmatched_reason: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.gold.stable_id,
            "source_text": self.gold.source_text,
            "gold_frame": self.gold.gold_frame,
            "graph": {
                "terms": self.matched_terms,
                "predicates": self.matched_predicates,
                "rules": self.matched_rules,
            },
            "matching": {
                "matched_rule_ids": [rule.get("id") for rule in self.matched_rules if rule.get("id")],
                "unmatched_reason": self.unmatched_reason,
            },
        }


JUDGE_SYSTEM_PROMPT = """You are a strict clinical graph evaluation judge.

Task: evaluate whether a v2 typed graph faithfully, completely, and computably represents the clinical decision semantics of the source recommendation.

Use the chain:
source recommendation -> entities -> predicates -> rule condition_dag -> action.

Compare against the provided human gold_frame, not against your own medical preference.

Scoring:
- 0 = completely wrong
- 1 = topic captured but unusable for decision support
- 2 = partially correct but misses key condition or action
- 3 = mostly correct with minor slot errors
- 4 = fully correct and ready for human review/execution preparation

Weighted total formula:
0.2 * Entity + 0.3 * Predicate + 0.3 * Rule Logic + 0.2 * Action.
Faithfulness is reported separately and should influence overall/risk.

Required checks:
- source -> graph: all key source semantics must appear in the graph.
- graph -> source: every condition, threshold, temporal constraint, and action must be grounded in source text or explicit context.
- Rule/action errors that change clinical decisions are high risk.
- Do not credit unsupported graph elements just because they are clinically plausible.
- Count scenarios should be aggregate(count) + compare.
- Action permission strength matters: consider/recommend/require/caution/avoid/contraindicate/stop are not interchangeable.

Return only a JudgeEvaluation tool call."""


def load_gold_jsonl(path: str | Path) -> List[GoldRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                records.append(GoldRecord.model_validate(data))
            except Exception as exc:
                raise ValueError(f"Invalid gold JSONL line {line_no}: {exc}") from exc
    return records


def write_gold_template(
    gen_dir: str | Path,
    output_path: str | Path,
    limit: Optional[int] = None,
) -> int:
    artifacts = PipelineArtifacts.load(gen_dir)
    rows = []
    for idx, prov in enumerate(artifacts.provenances[: limit or None], start=1):
        quote = prov.get("quote") or prov.get("source_text") or ""
        candidate_rule_ids = [
            rule.get("id")
            for rule in artifacts.rules
            if _rule_has_source_quote(rule, quote)
        ]
        rows.append(
            {
                "recommendation_id": f"rec.{idx:04d}",
                "source": prov.get("source"),
                "source_text": quote,
                "candidate_rule_ids": candidate_rule_ids,
                "gold_frame": {
                    "population": None,
                    "conditions": [],
                    "action": {
                        "subject": None,
                        "permission": None,
                    },
                    "timing": None,
                    "exceptions": [],
                    "strength": None,
                    "evidence": prov.get("evidence_level"),
                },
            }
        )

    _write_jsonl(output_path, rows)
    return len(rows)


def build_evaluation_context(gold: GoldRecord, artifacts: PipelineArtifacts) -> EvaluationContext:
    matched_rules = _match_rules(gold, artifacts)
    unmatched_reason = None if matched_rules else "No rule matched by rule_id/source_text."

    predicate_ids = set()
    term_ids = set()
    for rule in matched_rules:
        predicate_ids.update(rule.get("input_predicates") or [])
        predicate_ids.update(_find_strings(rule.get("condition_dag"), prefix="pred."))
        action = rule.get("action") or {}
        term_ids.update(action.get("subjects") or [])

    predicates_by_id = artifacts.predicates_by_id
    matched_predicates = [predicates_by_id[pred_id] for pred_id in predicate_ids if pred_id in predicates_by_id]
    for pred in matched_predicates:
        term_ids.add(pred.get("entity"))
        term_ids.update(pred.get("dependencies") or [])
        for value in _find_strings(pred, prefix=("cond.", "meas.", "obs.", "proc.", "med.")):
            term_ids.add(value)

    terms_by_id = artifacts.terms_by_id
    matched_terms = [terms_by_id[term_id] for term_id in term_ids if term_id in terms_by_id]

    if not matched_rules:
        matched_terms = _source_related_items(artifacts.terms + artifacts.med_terms, gold.source_text, limit=20)
        matched_predicates = _source_related_items(artifacts.predicates, gold.source_text, limit=20)

    return EvaluationContext(
        gold=gold,
        matched_rules=[_compact_rule(rule) for rule in matched_rules],
        matched_predicates=[_compact_predicate(pred) for pred in matched_predicates],
        matched_terms=[_compact_term(term) for term in matched_terms],
        unmatched_reason=unmatched_reason,
    )


def evaluate_gold_records(
    gold_records: List[GoldRecord],
    artifacts: PipelineArtifacts,
    output_dir: str | Path,
    llm_config: LLMConfig,
    max_items: Optional[int] = None,
    dry_run: bool = False,
    overwrite: bool = True,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    contexts_path = out / "evaluation_contexts.jsonl"
    results_path = out / "evaluation_results.jsonl"

    selected = gold_records[: max_items or None]
    contexts = [build_evaluation_context(record, artifacts) for record in selected]
    _write_jsonl(contexts_path, [ctx.to_payload() for ctx in contexts])

    if dry_run:
        return {
            "mode": "dry_run",
            "items": len(contexts),
            "contexts_path": str(contexts_path),
        }

    if not llm_config.api_key:
        raise RuntimeError("LLM_API_KEY is required unless --dry-run or --write-gold-template is used.")

    if overwrite and results_path.exists():
        results_path.unlink()

    client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
    results = []
    for idx, context in enumerate(contexts, start=1):
        print(f"[eval] judging {idx}/{len(contexts)} {context.gold.stable_id}")
        evaluation, raw = judge_context(client, llm_config, context)
        row = {
            "recommendation_id": context.gold.stable_id,
            "matched_rule_ids": [rule.get("id") for rule in context.matched_rules if rule.get("id")],
            "evaluation": evaluation.model_dump(exclude_none=True),
            "raw": raw,
        }
        results.append(row)
        _append_jsonl(results_path, row)

    summary = summarize_results(results, output_dir=out, model=llm_config.model)
    _save_json(out / "summary.json", summary)
    _save_markdown_report(out / "evaluation_report.md", summary, results)
    return summary


def judge_context(
    client: OpenAI,
    llm_config: LLMConfig,
    context: EvaluationContext,
) -> Tuple[JudgeEvaluation, Dict[str, Any]]:
    tool = _pydantic_to_function_schema(JudgeEvaluation)
    payload = context.to_payload()
    user_content = json.dumps(payload, ensure_ascii=False, indent=2)
    response = client.chat.completions.create(
        model=llm_config.model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "JudgeEvaluation"}},
        temperature=llm_config.temperature,
    )
    choice = response.choices[0]
    raw = {
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump() if response.usage else None,
    }

    args_str = None
    if choice.message.tool_calls:
        args_str = choice.message.tool_calls[0].function.arguments
    elif choice.message.content:
        args_str = _extract_json_object(choice.message.content)

    if not args_str:
        raise ValueError("Judge response did not contain tool arguments or JSON content.")

    try:
        data = json.loads(args_str)
        parsed = JudgeEvaluation.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Judge output parse/validation failed: {exc}; raw={args_str[:1000]}") from exc

    raw["arguments"] = data
    return parsed, raw


def summarize_results(results: List[Dict[str, Any]], output_dir: Path, model: str) -> Dict[str, Any]:
    score_keys = ["entity", "predicate", "rule_logic", "action", "faithfulness", "overall", "weighted_total"]
    totals = {key: 0.0 for key in score_keys}
    counts = {key: 0 for key in score_keys}
    verdicts = Counter()
    error_types = Counter()
    clinical_risk = Counter()

    for row in results:
        evaluation = row["evaluation"]
        scores = evaluation.get("scores", {})
        for key in score_keys:
            if key in scores and scores[key] is not None:
                totals[key] += float(scores[key])
                counts[key] += 1
        verdicts[evaluation.get("verdict", "unknown")] += 1
        clinical_risk[evaluation.get("clinical_risk", "unknown")] += 1
        for error in evaluation.get("errors", []):
            error_types[error.get("type", "unknown")] += 1

    averages = {
        key: round(totals[key] / counts[key], 3) if counts[key] else None
        for key in score_keys
    }
    return {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "output_dir": str(output_dir),
        "total_items": len(results),
        "average_scores": averages,
        "verdict_counts": dict(verdicts),
        "clinical_risk_counts": dict(clinical_risk),
        "error_type_counts": dict(error_types),
    }


def llm_config_from_env(temperature: float = 0.0) -> LLMConfig:
    """评估脚本用固定 temperature；其余与 pipeline LLM 环境变量约定一致。"""
    return LLMConfig.from_env(temperature=temperature)


def _match_rules(gold: GoldRecord, artifacts: PipelineArtifacts) -> List[Dict[str, Any]]:
    rules_by_id = artifacts.rules_by_id
    requested = [rules_by_id[rule_id] for rule_id in gold.requested_rule_ids if rule_id in rules_by_id]
    if requested:
        return requested

    source_text = gold.source_text
    matched = [rule for rule in artifacts.rules if _rule_has_source_quote(rule, source_text)]
    if matched:
        return matched

    if gold.recommendation_id:
        matched = [rule for rule in artifacts.rules if gold.recommendation_id in _json_text(rule)]
        if matched:
            return matched
    return []


def _rule_has_source_quote(rule: Dict[str, Any], source_text: str) -> bool:
    target = _normalize_text(source_text)
    if not target:
        return False
    for prov in rule.get("provenance") or []:
        quote = _normalize_text(prov.get("quote") or prov.get("source_text") or "")
        if quote and (quote in target or target in quote):
            return True
    return False


def _source_related_items(items: List[Dict[str, Any]], source_text: str, limit: int) -> List[Dict[str, Any]]:
    target = _normalize_text(source_text)
    scored = []
    for item in items:
        text = _normalize_text(_json_text(item))
        if not text:
            continue
        score = len(set(target) & set(text)) / max(len(set(target)), 1)
        if source_text and (target in text or any(tok in text for tok in _tokens(target))):
            score += 1.0
        if score > 0.15:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def _compact_term(term: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(term, [
        "id", "name", "label", "type", "clinical_entity", "concept", "value_domain", "unit",
        "value_set_binding", "code_bindings", "data_bindings", "fhir_binding_hint",
        "omop_binding_hint", "normalization_confidence", "source_evidence",
    ])


def _compact_predicate(pred: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(pred, [
        "id", "name", "description", "source_text", "entity", "entity_type", "aspect",
        "input_shape", "reduction", "return_type", "final_output_type", "temporal_scope",
        "library_function", "value_set_binding", "code_binding", "unit", "quantity_semantics",
        "range_spec", "retrieve", "filters", "extract", "compare", "null_policy",
        "evidence", "source_span", "dependencies",
    ])


def _compact_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    return _pick(rule, [
        "id", "label", "input_predicates", "condition_dag", "boolean_root", "action",
        "scope", "priority", "missing_data_policy", "provenance", "output_assembly",
        "condition",
    ])


def _pick(obj: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {key: obj[key] for key in keys if key in obj and obj[key] is not None}


def _find_strings(obj: Any, prefix: str | Tuple[str, ...]) -> List[str]:
    found = []
    prefixes = (prefix,) if isinstance(prefix, str) else prefix
    if isinstance(obj, str):
        if obj.startswith(prefixes):
            found.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            found.extend(_find_strings(value, prefixes))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_find_strings(value, prefixes))
    return found


def _pydantic_to_function_schema(model_cls) -> dict:
    schema = model_cls.model_json_schema()
    if "$defs" in schema:
        defs = schema.pop("$defs")

        def resolve_refs(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"].split("/")[-1]
                    return resolve_refs(defs.get(ref_path, obj))
                return {k: resolve_refs(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [resolve_refs(item) for item in obj]
            return obj

        schema = resolve_refs(schema)

    return {
        "type": "function",
        "function": {
            "name": model_cls.__name__,
            "description": model_cls.__doc__ or model_cls.__name__,
            "parameters": schema,
        },
    }


def _write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _save_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _load_json(path: str | Path, default: Any) -> Any:
    if not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_markdown_report(path: str | Path, summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    lines = [
        "# Source-Grounded Semantic Equivalence Evaluation",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Model: `{summary['model']}`",
        f"- Total items: `{summary['total_items']}`",
        "",
        "## Average Scores",
        "",
        "| Layer | Score |",
        "|---|---:|",
    ]
    for key, value in summary["average_scores"].items():
        lines.append(f"| {key} | {value if value is not None else 'NA'} |")
    lines.extend(["", "## Verdict Counts", "", "```json", json.dumps(summary["verdict_counts"], ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## High-Risk / Major Items", ""])
    for row in results:
        evaluation = row["evaluation"]
        if evaluation.get("clinical_risk") == "high" or evaluation.get("verdict") in {"major_errors", "wrong", "unmatched"}:
            lines.append(f"- `{row['recommendation_id']}` {evaluation.get('verdict')} risk={evaluation.get('clinical_risk')}: {evaluation.get('rationale', '')}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _extract_json_object(text: str) -> Optional[str]:
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _tokens(text: str) -> List[str]:
    return [tok for tok in re.split(r"[\W_]+", text.lower()) if len(tok) >= 3]


def _json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _stable_text_id(text: str) -> str:
    import hashlib

    return "rec." + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
