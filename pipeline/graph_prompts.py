"""Static prompts and library function listing for extraction graphs."""
from .library_registry import get_registered_library_function_ids

# ============ 系统提示词 ============

REGISTERED_LIBRARY_FUNCTIONS_TEXT = "\n".join(
    f"- {function_id}" for function_id in get_registered_library_function_ids()
)

RECOMMENDATION_PROMPT = """
你是一名临床专家，给定以下的临床指南文本，你的核心任务是：**从中精确识别并提取出可作为临床决策指令的“行动建议”条目。**

### 一、 核心识别标准：什么是“行动建议”？
一条合格的“行动建议”必须同时满足以下三个条件：
1.  **有明确的目标人群**：清晰定义了“谁”（例如：T2DM患者、CKD 3-4期患者、使用SGLT2i后eGFR下降≥30%的患者）。
2.  **有明确的临床情境或前提条件**：说明了“在什么情况下”（例如：单药治疗3个月未达标、起始胰岛素治疗时、eGFR<30 ml/min时）。
3.  **有明确的、可执行的行动指令**：包含一个表示“做什么”的动词，通常是**建议、推荐、考虑、应、可、需、避免、禁忌、起始、加用、停用、调整剂量至...** 等。

### 二、 关键区分：什么**不是**“行动建议”？（应被过滤）
请特别注意过滤以下类型的陈述，**它们不应被提取**：
-   **背景信息/事实陈述**：描述疾病、药物特性或现状，但没有给出针对性的行动指令。
    -   *示例*：“β受体阻滞剂在该类患者中耐受性更好。”（描述特性）
    -   *示例*：“SGLT2i可降低HFrEF患者的心血管死亡风险。”（陈述获益）
    -   *示例*：“老年T2DM患者常存在多种并发症。”（描述现状）
-   **治疗目标/原则声明**：说明了治疗目标或一般原则，但没有给出具体如何实现的操作指令。
    -   *示例*：“治疗需个体化。”
    -   *示例*：“应综合管理血糖、血压、血脂。”
-   **不完整的建议片段**：只说了“什么药”，没说“给谁用”或“在什么情况下用”。
    -   *处理*：仅当能用**用户正文中紧邻的原文句子**按第三节规则拼接补全时方可输出；否则**不提取**。

### 三、 提取与格式化要求
1.  **完整性**：一个完整的`quote`必须是一个能独立传达临床决策信息的句子或句群，**明确包含“目标人群+情境条件+行动指令”**。若单句不足，仅允许从**紧邻的后续或前文**拼接**原文已出现的句子**以补全，且拼接后的字符串仍须全部来自原文逐字连续片段（允许中间省略无关句，但**不得**插入原文未出现的词）。
2.  **原子性**：一条`quote`应对应一个独立的临床决策。如果原文是复合句（如“如果A，则做X；如果B，则做Y”），应拆分为两条独立的`quote`。
3.  **忠于原文（硬约束，违反则宁可不输出该条）**：
    - `quote` 必须可由用户消息中的正文**逐字复现**（连续子串，或按上条规则由**相邻原文句**顺序拼接而成的子串）；**禁止**同义改写、概括、翻译或换用未在原文出现的表述。
    - **语种一致**：用户正文以何种语言书写，`quote` 必须与该处正文**同一语种**；**严禁**将英文译成中文、中文译成英文，或中英混写（除非原文本身如此）。
    - **禁止**凭医学常识补写原文未写明的药物、剂量或操作；`source` 若未知可填指南通用名或从用户提供的标题行合理推断，但 `quote` 仍须严格来自用户正文。
4.  **表格处理**：如为表格，视每一行为一条独立的推荐意见进行提取，确保每行信息都构成一个完整的“行动建议”，且每行 `quote` 仍须为该行的原文摘录。

### 四、 输出格式
对于每一个识别出的“行动建议”，按以下格式输出：
```json
{
    "source": "指南名称（年份版）",
    "quote": "完整且包含目标人群、条件和行动指令的推荐意见文本。",
    "recommendation_grade": "推荐等级（如I, IIa, IIb, III，若无则填‘None‘）",
    "evidence_level": "证据等级（如A, B, C，若无则填‘None‘）"
}
"""

TERMS_AND_MEDS_PROMPT = """你是一名临床信息模型专家。从临床指南中抽取术语。

必须输出 TermExtractionResult。最多抽取 10 个非药物术语和 10 个药物术语，优先抽取规则条件、动作对象和数值阈值相关术语。

每个非药物 term 必须包含：
- id: 稳定语义 ID，如 cond.t2dm、meas.egfr、obs.diarrhea、proc.surgery
- name: 英文标准医学名
- label: measures / conditions / procedures / observations
- type: value domain，优先使用 Bool、Quantity、Code、DateTime、Interval、Enum
- clinical_entity: canonical entity id，通常与 id 相同
- concept: 标准化概念名
- value_set_binding 或 code_bindings: 只能作为 candidate。只有原文明确给出 code/ValueSet 时才填写具体 code；未知时 type=Unknown 且 confidence 低，不要凭记忆发明标准 code
- data_bindings / fhir_binding_hint / omop_binding_hint: 只能作为 candidate hint，最多给出 resource/table 级线索；不要输出 verified_binding
- binding_status: 使用 candidate 或 unresolved。verified_binding 由后处理 binding resolver 填充
- source_evidence: 原文证据，保留 source_text
- normalization_confidence: 0 到 1

每个药物 med_term 必须包含：
- id: med.xxx 或 med.class.xxx
- name: 英文通用名或药物类别名
- clinical_entity / concept
- class / subclass: 药物类层级；类别本身可为 null
- value_set_binding 或 code_bindings: RxNorm/ATC/OMOP/Unknown，均为 candidate；不要凭记忆发明药品 code
- data_bindings / fhir_binding_hint / omop_binding_hint: candidate hint
- binding_status: candidate 或 unresolved；verified_binding 留空
- source_evidence / normalization_confidence

## 命名规范
- measures: meas.egfr, meas.hba1c, meas.body_temperature
- conditions: cond.t2dm, cond.ckd, cond.hf, cond.ascvd
- procedures: proc.surgery, proc.ct_abdomen
- observations: obs.diarrhea, obs.smoking_history
- drug class: med.class.sglt2i
- drug ingredient: med.empagliflozin

## 示例
{
  "terms": [
    {
      "id": "meas.egfr",
      "name": "Estimated glomerular filtration rate",
      "label": "measures",
      "type": "Quantity",
      "clinical_entity": "meas.egfr",
      "concept": "Estimated glomerular filtration rate",
      "value_domain": "Quantity",
      "unit": "mL/min/1.73m2",
      "value_set_binding": {"type": "Unknown", "name": "Estimated glomerular filtration rate", "confidence": 0.2},
      "data_bindings": {"FHIR": {"resource": "Observation"}, "OMOP": {"table": "measurement"}},
      "fhir_binding_hint": {"resource": "Observation"},
      "omop_binding_hint": {"table": "measurement"},
      "binding_status": "candidate",
      "source_evidence": [{"source_text": "eGFR >= 20"}],
      "normalization_confidence": 0.86
    }
  ],
  "med_terms": [
    {
      "id": "med.class.sglt2i",
      "name": "Sodium-glucose cotransporter 2 inhibitors",
      "class": null,
      "subclass": null,
      "clinical_entity": "med.class.sglt2i",
      "concept": "SGLT2 inhibitor drug class",
      "value_set_binding": {"type": "ValueSet", "name": "SGLT2 inhibitors", "confidence": 0.45},
      "data_bindings": {"FHIR": {"resource": "MedicationStatement"}, "OMOP": {"table": "drug_exposure"}},
      "binding_status": "candidate",
      "source_evidence": [{"source_text": "SGLT2i"}],
      "normalization_confidence": 0.9
    }
  ]
}

肝肾功能不全/受损以及分级相关的标准术语优先复用：
- cond.hepatic_impairment: Hepatic Functional Severity (Child-Pugh)/Liver Disease Diagnosis/History
- cond.renal_impairment: Renal Impairment
"""


PREDICATES_PROMPT = """You are a clinical CQL-like typed graph extraction agent.

Given guideline quotes plus available v2 terms and medications, extract every atomic clinical precondition as a complete v2 typed predicate schema. Do not output legacy formal_definition text as the core representation.

Must call PredicateExtractionBatch with a "predicates" list. Every predicate must include:
- id, name, description, source_text
- entity, entity_type, aspect
- input_shape
- reduction {operator, output_type}
- return_type and final_output_type
- temporal_scope {mode, lookback/date_fallback/time_paths when applicable}
- data_binding / retrieve / extract / filters / compare as applicable
- library_function references when a standard helper is used
- value_set_binding or code_binding only when inherited from available terms or explicitly present in source. Treat LLM-provided term bindings as candidate; do not invent executable codes or FHIR paths.
- unit and quantity_semantics for numeric predicates
- range_spec for quantity_range predicates or predicates whose id ends with .range
- null_policy
- evidence and source_span
- dependencies containing referenced term ids

## Canonical v2 predicate design
- exists: List<Resource> -> Bool may be internal predicate reduction.
- most_recent / max / min / extremum may be internal predicate reduction for common clinical observations.
- count must not be collapsed to a Bool predicate when the rule compares a count. Prefer a predicate returning List<Resource>, then a Rule DAG aggregate(count) + compare.
- A simple threshold such as eGFR >= 20 may be a Bool predicate with reduction most_recent and compare.
- Active/current concepts should use library_function such as lib.fhir.condition.active, lib.fhir.filter_by_status, or lib.fhir.filter_by_lookback.
- Do not use temporal_scope.mode="most_recent_all_time"; encode that as reduction.operator="most_recent" with temporal_scope.mode="all_time".
- Quantity range predicates must set final_output_type="Enum" and include range_spec.intervals with unique interval ids and legal bounds.

## Standard library function IDs
Use only ids from this registry:
""" + REGISTERED_LIBRARY_FUNCTIONS_TEXT + """

## Examples
{
  "predicates": [
    {
      "id": "pred.cond.t2dm.exists",
      "name": "Type 2 diabetes mellitus exists",
      "description": "Patient has type 2 diabetes mellitus.",
      "source_text": "patients with T2DM",
      "entity": "cond.t2dm",
      "entity_type": "condition",
      "aspect": "existence",
      "input_shape": "List<Condition>",
      "reduction": {"operator": "exists", "output_type": "Bool"},
      "return_type": "Bool",
      "final_output_type": "Bool",
      "temporal_scope": {"mode": "all_time"},
      "data_binding": {"FHIR": {"resource": "Condition", "code_path": "Condition.code"}, "OMOP": {"table": "condition_occurrence"}},
      "retrieve": {"resource": "Condition", "code_binding": "cond.t2dm"},
      "library_function": [],
      "value_set_binding": {"type": "ValueSet", "name": "Type 2 diabetes mellitus", "confidence": 0.7},
      "unit": null,
      "quantity_semantics": {},
      "compare": null,
      "null_policy": "unknown",
      "evidence": [{"source_text": "patients with T2DM"}],
      "source_span": {"source_text": "patients with T2DM"},
      "dependencies": ["cond.t2dm"]
    },
    {
      "id": "pred.meas.egfr.value.ge.20",
      "name": "Most recent eGFR >= 20",
      "description": "Most recent eGFR value is at least 20 mL/min/1.73m2.",
      "source_text": "eGFR >= 20",
      "entity": "meas.egfr",
      "entity_type": "observation",
      "aspect": "quantity",
      "input_shape": "List<Observation>",
      "reduction": {"operator": "most_recent", "output_type": "Quantity"},
      "return_type": "Bool",
      "final_output_type": "Bool",
      "temporal_scope": {"mode": "all_time", "date_fallback": ["effectiveDateTime", "effectivePeriod.end", "effectivePeriod.start", "issued"]},
      "data_binding": {"FHIR": {"resource": "Observation", "value_path": "Observation.valueQuantity"}, "OMOP": {"table": "measurement", "value_path": "value_as_number"}},
      "retrieve": {"resource": "Observation", "code_binding": "meas.egfr"},
      "extract": {"path": "valueQuantity", "type": "Quantity", "unit": "mL/min/1.73m2"},
      "compare": {"operator": "ge", "value": 20, "unit": "mL/min/1.73m2"},
      "library_function": ["lib.fhir.most_recent"],
      "unit": "mL/min/1.73m2",
      "quantity_semantics": {"unit": "mL/min/1.73m2", "comparator": "ge", "value": 20},
      "null_policy": "unknown",
      "evidence": [{"source_text": "eGFR >= 20"}],
      "source_span": {"source_text": "eGFR >= 20"},
      "dependencies": ["meas.egfr"]
    }
  ]
}
"""


RULES_PROMPT = """你是一名规则抽取专家。输入包括 quote ids、可用 typed predicates 和药物术语。你的任务是把 quote 行动指令转成原子化 ClinicalRule，并用 condition_dag 表达条件。

必须调用 SubmitSimplifiedRules 工具。不得把 legacy condition string 作为核心表达；核心条件必须是 condition_dag。

## Rule DAG 必须支持并正确使用
- predicate_ref: 引用一个谓词作为 Bool/Quantity/List<Resource> 输入
- combine: boolean all/any/not，对应 operator and/or/not
- compare: 对 Integer/Quantity/Enum/DateTime 做比较，输出 Bool
- aggregate: 对 List<T> 做 count/max/min/most_recent；尤其 count 场景必须 aggregate(count) + compare
- coalesce: 多个同类型候选值取第一个非 null
- temporal_relation: before/after/during/overlaps/in_interval 等
- typed intermediate values: 每个节点必须有 return_type
- action/output assembly: 若 quote 是输出集合组装，必须放在 rule.output_assembly 或 action.output，不要塞进 condition_dag.root

## DAG 校验约束
- condition_dag.root 必须指向 nodes 中存在的节点，且该节点 return_type 必须是 Bool
- boolean_root 应与 condition_dag.root 一致；若不同，也必须指向存在的 Bool 节点
- 节点 id 不得重复
- combine 节点必须提供非空 inputs
- compare 节点必须提供 left 和 right
- aggregate(count) 节点必须 return_type=Integer
- coalesce 节点至少 2 个 inputs
- range_membership 必须提供 input 和 value，且 return_type=Bool
- temporal_relation 必须 return_type=Bool，operator 使用 before/after/during/overlaps/starts/ends/meets/same_or_before/same_or_after/within/in_interval
- aggregate(max|min|extremum) 默认 return_type=Quantity
- library_function 节点必须引用已注册 library function id

## Count 规范
当规则语义是 “次数 > n / 至少 n 次 / Count(...)”：
1. input_predicates 引用返回 List<Resource> 的 predicate
2. condition_dag 节点先 aggregate operator=count return_type=Integer
3. 再 compare operator=gt/ge/eq return_type=Bool
不要把 Count > n 压成单个 exists Bool 谓词。

## 已注册 library function id
condition_dag 的 library_function 节点只能使用以下 id：
""" + REGISTERED_LIBRARY_FUNCTIONS_TEXT + """

## Action 规范
action.subjects 必须使用标准药物/操作/输出对象 id。
permission 必须为 recommend/require/allow/consider/caution/avoid/contraindicate/continue/stop/reduce_dose/increase_dose/start_low_dose/max_dose_limit/titrate/maintain_dose。
保留 strength、intent、timing、monitoring、requirements、conflict_profile 等可用语义。

## 输出字段
每条规则必须包含：
- id, label
- input_predicates
- condition_dag {nodes, root}
- boolean_root
- missing_data_policy
- action
- priority {recommendation_grade, evidence_level, source_date, jurisdiction}; 未知可留空
- output_assembly: 可选；operator 支持 union_except_null, union, except_null, message_list
- source_ids: 单个 quote id，如 q1

## 示例：AND + OR
{
  "id": "rule.dm_with_ascvd_or_high_risk_sglt2i_recommend",
  "label": "T2DM with ASCVD or high ASCVD risk recommends SGLT2i",
  "input_predicates": ["pred.cond.t2dm.exists", "pred.cond.ascvd.exists", "pred.cond.ascvd.risk.eq.High"],
  "condition_dag": {
    "nodes": [
      {"id": "N1", "type": "combine", "operator": "or", "inputs": ["pred.cond.ascvd.exists", "pred.cond.ascvd.risk.eq.High"], "return_type": "Bool"},
      {"id": "ROOT", "type": "combine", "operator": "and", "inputs": ["pred.cond.t2dm.exists", "N1"], "return_type": "Bool"}
    ],
    "root": "ROOT"
  },
  "boolean_root": "ROOT",
  "missing_data_policy": "propagate_unknown",
  "action": {"subjects": ["med.class.sglt2i"], "permission": "recommend", "strength": "strong", "intent": "initiate_or_continue", "requirements": []},
  "priority": {"recommendation_grade": null, "evidence_level": null},
  "source_ids": "q1"
}

## 示例：count + compare
{
  "id": "rule.cdi_diarrhea_count_24h",
  "label": "At least three diarrhea observations in 24 hours",
  "input_predicates": ["pred.obs.diarrhea.list_24h"],
  "condition_dag": {
    "nodes": [
      {"id": "N1", "type": "aggregate", "operator": "count", "input": "pred.obs.diarrhea.list_24h", "return_type": "Integer"},
      {"id": "ROOT", "type": "compare", "operator": "ge", "left": "N1", "right": 3, "return_type": "Bool"}
    ],
    "root": "ROOT"
  },
  "boolean_root": "ROOT",
  "missing_data_policy": "propagate_unknown",
  "action": {"subjects": ["output.cdi.high_clinical_suspicion"], "permission": "recommend", "intent": "derive_state", "requirements": []},
  "priority": {"recommendation_grade": null, "evidence_level": null},
  "source_ids": "q1"
}
"""



