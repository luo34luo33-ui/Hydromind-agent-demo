import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple


def get_knowledge_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "knowledge"


class ConstraintReasoner:
    """约束推理器 - 验证机制组合的物理约束"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._load_constraints()
        self._initialized = True

    def _load_constraints(self):
        """加载约束定义"""
        constraints_file = get_knowledge_dir() / "constraints.json"
        if constraints_file.exists():
            with open(constraints_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.constraints = data.get("constraints", [])
                self.compatibility_rules = data.get("compatibility_rules", [])
        else:
            self.constraints = []
            self.compatibility_rules = []

    def check_compatibility(
        self,
        runoff_mechanism: str,
        routing_mechanism: str
    ) -> Tuple[bool, str]:
        """
        检查产流机制和汇流机制的兼容性
        
        返回:
            (is_compatible: bool, reason: str)
        """
        for rule in self.compatibility_rules:
            if rule.get("from") == runoff_mechanism and rule.get("to") == routing_mechanism:
                if rule.get("compatible", False):
                    return True, rule.get("reason", "Compatible")
                else:
                    return False, rule.get("reason", "Incompatible")

        return True, "No specific rule, assume compatible"

    def validate_blueprint(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证机制蓝图的合法性
        
        返回:
            {
                "valid": bool,
                "violations": List[str],
                "warnings": List[str]
            }
        """
        violations = []
        warnings = []

        runoff = blueprint.get("runoff_mechanism", "")
        routing = blueprint.get("routing_mechanism", "")
        patterns = blueprint.get("patterns", [])
        constraints = blueprint.get("constraints", [])

        compatible, reason = self.check_compatibility(runoff, routing)
        if not compatible:
            violations.append(f"Compatibility violation: {runoff} + {routing} - {reason}")

        for constraint in self.constraints:
            if constraint.get("enforcement") == "hard":
                if constraint.get("constraint_id") not in constraints:
                    warnings.append(f"Missing recommended constraint: {constraint.get('constraint_id')}")

        has_mass_conservation = "mass_conservation" in constraints or "mass_conservation" in patterns
        if not has_mass_conservation:
            warnings.append("建议添加水量守恒约束")

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings
        }

    def get_suitable_mechanisms(
        self,
        climate: str,
        area: float = None
    ) -> Dict[str, List[str]]:
        """
        根据流域特征推荐适用机制
        
        返回:
            {
                "runoff": [...],
                "routing": [...],
                "evap": [...]
            }
        """
        climate_lower = climate.lower()
        result = {"runoff": [], "routing": [], "evap": []}

        if "humid" in climate_lower or "湿润" in climate_lower or "南方" in climate_lower:
            result["runoff"] = ["saturation_excess", "soil_moisture_accounting"]
            result["routing"] = ["linear_reservoir", "cascade"]
            result["evap"] = ["layered_et"]
        elif "arid" in climate_lower or "干旱" in climate_lower or "北方" in climate_lower:
            result["runoff"] = ["infiltration_excess"]
            result["routing"] = ["linear_reservoir"]
            result["evap"] = ["soil_et"]
        else:
            result["runoff"] = ["soil_moisture_accounting", "saturation_excess"]
            result["routing"] = ["linear_reservoir"]
            result["evap"] = ["soil_et"]

        return result

    def get_required_constraints(self, mechanisms: List[str]) -> List[str]:
        """获取指定机制组合所需的约束列表"""
        required = []

        if "saturation_excess" in mechanisms or "infiltration_excess" in mechanisms:
            required.append("runoff_nonnegative")
            required.append("runoff_bounded_by_water")

        if any(m in mechanisms for m in ["linear_reservoir", "cascade", "nash_routing"]):
            required.append("mass_conservation")
            required.append("storage_nonnegative")

        if "layered_et" in mechanisms:
            required.append("et_nonnegative")
            required.append("et_limited_by_pet")
            required.append("layered_et_priority")

        return list(set(required))


def get_reasoner() -> ConstraintReasoner:
    """获取约束推理器实例"""
    return ConstraintReasoner()