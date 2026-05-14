import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from .mechanism_registry import get_registry, MechanismTemplate


def get_knowledge_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "knowledge"


class KGRAG:
    """知识图谱增强的RAG检索引擎"""

    def __init__(self):
        self.registry = get_registry()
        self._load_patterns()
        self._load_constraints()

    def _load_patterns(self):
        """加载模式定义"""
        patterns_file = get_knowledge_dir() / "patterns.json"
        if patterns_file.exists():
            with open(patterns_file, "r", encoding="utf-8") as f:
                self.patterns_data = json.load(f)
        else:
            self.patterns_data = {"patterns": []}

    def _load_constraints(self):
        """加载约束定义"""
        constraints_file = get_knowledge_dir() / "constraints.json"
        if constraints_file.exists():
            with open(constraints_file, "r", encoding="utf-8") as f:
                self.constraints_data = json.load(f)
        else:
            self.constraints_data = {"constraints": [], "compatibility_rules": []}

    def retrieve_mechanisms(
        self,
        basin_attributes: str,
        user_requirements: str = ""
    ) -> Dict[str, Any]:
        """
        根据流域属性检索匹配的机制组合
        
        返回:
            {
                "runoff_mechanisms": [...],
                "routing_mechanisms": [...],
                "evap_mechanisms": [...],
                "patterns": [...],
                "reasoning": str
            }
        """
        basin_lower = basin_attributes.lower()
        
        runoff_candidates = []
        routing_candidates = []
        evap_candidates = []

        if any(kw in basin_lower for kw in ["湿润", "humid", "南方", "subhumid"]):
            runoff_candidates = [
                self.registry.get_mechanism("saturation_excess"),
                self.registry.get_mechanism("soil_moisture_accounting")
            ]
            routing_candidates = [
                self.registry.get_mechanism("linear_reservoir"),
                self.registry.get_mechanism("cascade")
            ]
            evap_candidates = [self.registry.get_mechanism("layered_et")]

        elif any(kw in basin_lower for kw in ["干旱", "arid", "北方", "semi_arid", "small"]):
            runoff_candidates = [self.registry.get_mechanism("infiltration_excess")]
            routing_candidates = [self.registry.get_mechanism("linear_reservoir")]
            evap_candidates = [self.registry.get_mechanism("soil_et")]

        else:
            runoff_candidates = [self.registry.get_mechanism("soil_moisture_accounting")]
            routing_candidates = [self.registry.get_mechanism("linear_reservoir")]
            evap_candidates = [self.registry.get_mechanism("soil_et")]

        runoff_candidates = [m for m in runoff_candidates if m is not None]
        routing_candidates = [m for m in routing_candidates if m is not None]
        evap_candidates = [m for m in evap_candidates if m is not None]

        patterns = []
        for m in runoff_candidates + routing_candidates:
            patterns.extend(m.patterns)
        patterns = list(set(patterns))

        reasoning = self._generate_reasoning(basin_attributes, runoff_candidates, routing_candidates)

        return {
            "runoff_mechanisms": [m.to_dict() for m in runoff_candidates],
            "routing_mechanisms": [m.to_dict() for m in routing_candidates],
            "evap_mechanisms": [m.to_dict() for m in evap_candidates],
            "patterns": patterns,
            "reasoning": reasoning
        }

    def _generate_reasoning(
        self,
        basin_attributes: str,
        runoff_mechs: List[MechanismTemplate],
        routing_mechs: List[MechanismTemplate]
    ) -> str:
        """生成机制选择推理"""
        basin_lower = basin_attributes.lower()
        reasoning_parts = []

        if any(kw in basin_lower for kw in ["湿润", "humid", "南方"]):
            reasoning_parts.append("湿润区优先选择蓄满产流机制")
        elif any(kw in basin_lower for kw in ["干旱", "arid"]):
            reasoning_parts.append("干旱区选择超渗产流机制")

        if runoff_mechs:
            reasoning_parts.append(f"产流: {runoff_mechs[0].name}")

        if routing_mechs:
            reasoning_parts.append(f"汇流: {routing_mechs[0].name}")

        return "; ".join(reasoning_parts)

    def get_template_path(self, mechanism_id: str) -> Optional[Path]:
        """获取机制模板文件路径"""
        mech = self.registry.get_mechanism(mechanism_id)
        if mech:
            template_path = mech.implementation_template
            if template_path:
                return get_knowledge_dir() / "templates" / template_path
        return None

    def get_compatible_routing(self, runoff_mechanism_id: str) -> List[str]:
        """获取与指定产流机制兼容的汇流机制"""
        mech = self.registry.get_mechanism(runoff_mechanism_id)
        if mech:
            return mech.compatible_routing
        return []


def get_kg_rag() -> KGRAG:
    """获取KG-RAG实例"""
    return KGRAG()