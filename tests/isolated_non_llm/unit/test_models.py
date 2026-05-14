"""
单元测试 - pipeline/models.py
测试目标:
- merge_by_id reducer 去重逻辑
- _to_models 类型转换
- Permission 枚举方法
- Pydantic 模型验证
"""
import pytest
import logging

from pipeline.models import (
    Term, MedicationTerm, Predicates, ClinicalRule, Provenance,
    ProvenanceCluster, TermLabel, Permission, Action,
    TermList, MedicationTermList, PredicatesList, ClinicalRuleList,
    merge_by_id, _to_models, merge_cluster_cache_updates
)


class TestMergeById:
    """merge_by_id reducer 测试"""

    def test_merge_by_id_deduplicates(self):
        """相同 id 的项应该被去重，后项覆盖前项"""
        existing = [
            Term(id="1", name="A", label=TermLabel.MEASURES, type="x"),
            Term(id="2", name="B", label=TermLabel.MEASURES, type="y"),
        ]
        new = [
            Term(id="1", name="A_updated", label=TermLabel.CONDITIONS, type="z"),
            Term(id="3", name="C", label=TermLabel.MEASURES, type="w"),
        ]
        result = merge_by_id(existing, new)

        assert len(result) == 3
        item_1 = next(t for t in result if t.id == "1")
        assert item_1.name == "A_updated"
        assert item_1.label == TermLabel.CONDITIONS
        assert any(t.id == "3" for t in result)

    def test_merge_by_id_preserves_all_unique(self):
        """所有 id 唯一时应保留全部"""
        existing = [Term(id="1", name="A", label=TermLabel.MEASURES, type="x")]
        new = [Term(id="2", name="B", label=TermLabel.MEASURES, type="y")]
        result = merge_by_id(existing, new)

        assert len(result) == 2

    def test_merge_by_id_empty_inputs(self):
        """空输入应返回空列表"""
        assert merge_by_id([], []) == []
        assert len(merge_by_id([Term(id="1", name="A", label=TermLabel.MEASURES, type="x")], [])) == 1
        assert len(merge_by_id([], [Term(id="1", name="A", label=TermLabel.MEASURES, type="x")])) == 1

    def test_merge_by_id_handles_missing_id(self, caplog):
        """没有 id 的项应该被跳过，记录警告"""
        caplog.set_level(logging.WARNING)

        class NoId:
            pass

        result = merge_by_id([NoId()], [])
        assert result == []
        assert "跳过没有有效 id" in caplog.text

    def test_merge_by_id_handles_none_id(self, caplog):
        """id 为 None 的项应该被跳过"""
        caplog.set_level(logging.WARNING)

        class WithNoneId:
            id = None

        result = merge_by_id([WithNoneId()], [])
        assert result == []

    def test_merge_by_id_with_medications(self):
        """测试药物术语的去重"""
        existing = [
            MedicationTerm(id="med.1", name="SGLT2i", drug_class="med.sglt2i")
        ]
        new = [
            MedicationTerm(id="med.1", name="SGLT2抑制剂", drug_class="med.sglt2i"),
            MedicationTerm(id="med.2", name="GLP-1RA", drug_class="med.glp1ra"),
        ]
        result = merge_by_id(existing, new)

        assert len(result) == 2
        item_1 = next(m for m in result if m.id == "med.1")
        assert item_1.name == "SGLT2抑制剂"


class TestToModels:
    """_to_models 类型转换测试"""

    def test_to_models_from_dicts(self):
        items = [
            {"id": "1", "name": "eGFR", "label": "measures", "type": "lab"},
            {"id": "2", "name": "HbA1c", "label": "measures", "type": "lab"},
        ]
        result = _to_models(items, Term)

        assert len(result) == 2
        assert all(isinstance(t, Term) for t in result)
        assert result[0].id == "1"

    def test_to_models_from_models(self):
        items = [Term(id="1", name="eGFR", label=TermLabel.MEASURES, type="lab")]
        result = _to_models(items, Term)

        assert len(result) == 1
        assert result[0] is items[0]

    def test_to_models_mixed_input(self):
        items = [
            Term(id="1", name="eGFR", label=TermLabel.MEASURES, type="lab"),
            {"id": "2", "name": "HbA1c", "label": "measures", "type": "lab"},
        ]
        result = _to_models(items, Term)

        assert len(result) == 2
        assert all(isinstance(t, Term) for t in result)

    def test_to_models_invalid_dict_skipped(self, caplog):
        caplog.set_level(logging.WARNING)
        items = [
            {"id": "1", "name": "eGFR", "label": "measures", "type": "lab"},
            {"invalid": "data"},
        ]
        result = _to_models(items, Term)

        assert len(result) == 1
        assert "无法将字典验证为" in caplog.text

    def test_to_models_empty_input(self):
        assert _to_models([], Term) == []
        assert _to_models(None, Term) == []


class TestMergeClusterCacheUpdates:
    """merge_cluster_cache_updates 测试"""

    def test_merge_updates(self):
        existing = {1: {"terms": ["a"], "rules": ["r1"]}}
        new = {1: {"terms": ["b"]}, 2: {"terms": ["c"]}}

        result = merge_cluster_cache_updates(existing, new)

        assert result[1]["terms"] == ["b"]
        assert result[1]["rules"] == ["r1"]
        assert result[2]["terms"] == ["c"]

    def test_merge_empty(self):
        assert merge_cluster_cache_updates({}, {}) == {}
        assert merge_cluster_cache_updates({1: {"a": 1}}, {}) == {1: {"a": 1}}


class TestPermissionEnum:
    """Permission 枚举方法测试"""

    def test_usage_permissions(self):
        permissions = Permission.usage_permissions()
        assert Permission.CONTRAINDICATE in permissions
        assert Permission.RECOMMEND in permissions
        assert Permission.REDUCE_DOSE not in permissions

    def test_dose_permissions(self):
        permissions = Permission.dose_permissions()
        assert Permission.REDUCE_DOSE in permissions
        assert Permission.TITRATE in permissions
        assert Permission.RECOMMEND not in permissions

    def test_is_restrictive(self):
        assert Permission.CONTRAINDICATE.is_restrictive()
        assert Permission.AVOID.is_restrictive()
        assert Permission.CAUTION.is_restrictive()
        assert not Permission.RECOMMEND.is_restrictive()
        assert not Permission.ALLOW.is_restrictive()

    def test_is_permissive(self):
        assert Permission.RECOMMEND.is_permissive()
        assert Permission.ALLOW.is_permissive()
        assert Permission.REQUIRE.is_permissive()
        assert not Permission.CONTRAINDICATE.is_permissive()

    def test_is_dose_adjustment(self):
        assert Permission.REDUCE_DOSE.is_dose_adjustment()
        assert Permission.TITRATE.is_dose_adjustment()
        assert not Permission.RECOMMEND.is_dose_adjustment()

    def test_priority(self):
        assert Permission.CONTRAINDICATE.priority() == 0
        assert Permission.ALLOW.priority() > Permission.AVOID.priority()

    def test_priority_order(self):
        order = Permission.priority_order()
        assert order[0] == Permission.CONTRAINDICATE
        assert order[-1] == Permission.REQUIRE


class TestTermModel:
    def test_create_term(self):
        term = Term(
            id="term.1",
            name="eGFR",
            label=TermLabel.MEASURES,
            type="lab",
            unit="ml/min/1.73m2"
        )
        assert term.id == "term.1"
        assert term.name == "eGFR"
        assert term.label == TermLabel.MEASURES
        assert term.unit == "ml/min/1.73m2"

    def test_term_without_unit(self):
        term = Term(
            id="term.1",
            name="心力衰竭",
            label=TermLabel.CONDITIONS,
            type="diagnosis"
        )
        assert term.unit is None


class TestMedicationTermModel:
    def test_create_medication(self):
        med = MedicationTerm(
            id="med.sglt2i",
            name="SGLT2i",
            **{"class": "med.sglt2_inhibitor"}
        )
        assert med.id == "med.sglt2i"
        assert med.drug_class == "med.sglt2_inhibitor"

    def test_medication_alias(self):
        med = MedicationTerm(**{
            "id": "med.1",
            "name": "Test",
            "class": "med.test"
        })
        assert med.drug_class == "med.test"


class TestProvenanceModel:
    def test_create_provenance(self):
        p = Provenance(
            source="指南",
            quote="建议使用 SGLT2i",
            recommendation_grade="I",
            evidence_level="A"
        )
        assert p.source == "指南"
        assert p.quote == "建议使用 SGLT2i"
        assert p.recommendation_grade == "I"

    def test_bucket_id_field(self):
        p = Provenance(source="x", quote="y", bucket_id=5)
        assert p.bucket_id == 5

    def test_bucket_id_default_none(self):
        p = Provenance(source="x", quote="y")
        assert p.bucket_id is None


class TestClinicalRuleModel:
    def test_create_rule(self):
        rule = ClinicalRule(
            id="rule.1",
            label="测试规则",
            input_predicates=["pred.cond.hf.exists"],
            condition_dag={
                "nodes": [
                    {"id": "ROOT", "type": "predicate_ref", "predicate_ref": "pred.cond.hf.exists", "return_type": "Bool"}
                ],
                "root": "ROOT",
            },
            boolean_root="ROOT",
            action=Action(
                subjects=["med.sglt2i"],
                permission=Permission.RECOMMEND,
                requirements=["监测肾功能"]
            ),
            provenance=[]
        )
        assert rule.id == "rule.1"
        assert rule.condition_dag.root == "ROOT"
        assert rule.action.permission == Permission.RECOMMEND
        assert "监测肾功能" in rule.action.requirements


class TestProvenanceClusterModel:
    def test_create_cluster(self, sample_provenances):
        cluster = ProvenanceCluster(
            cluster_id=0,
            provenances=sample_provenances,
            texts_formatted=[f"Quote: {p.quote}" for p in sample_provenances]
        )
        assert cluster.cluster_id == 0
        assert len(cluster.provenances) == 3


class TestListWrappers:
    def test_term_list(self, sample_terms):
        wrapper = TermList(items=sample_terms)
        assert len(wrapper.items) == 3
        assert all(isinstance(t, Term) for t in wrapper.items)

    def test_medication_term_list(self, sample_med_terms):
        wrapper = MedicationTermList(items=sample_med_terms)
        assert len(wrapper.items) == 2

    def test_predicates_list(self, sample_predicates):
        wrapper = PredicatesList(items=sample_predicates)
        assert len(wrapper.items) == 2

    def test_clinical_rule_list(self, sample_rules):
        wrapper = ClinicalRuleList(items=sample_rules)
        assert len(wrapper.items) == 2


class TestTermLabelEnum:
    def test_term_label_values(self):
        assert TermLabel.MEASURES.value == "measures"
        assert TermLabel.CONDITIONS.value == "conditions"
        assert TermLabel.PROCEDURES.value == "procedures"
        assert TermLabel.OBSERVATIONS.value == "observations"

    def test_term_label_from_string(self):
        assert TermLabel("measures") == TermLabel.MEASURES
        assert TermLabel("conditions") == TermLabel.CONDITIONS
