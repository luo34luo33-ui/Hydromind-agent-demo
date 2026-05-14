import json
from pathlib import Path
from typing import Dict, List, Optional, Any


def get_knowledge_dir() -> Path:
    """获取knowledge目录路径"""
    return Path(__file__).resolve().parent.parent.parent / "knowledge"


class MechanismTemplate:
    """机制模板类 - 原子级水文过程"""

    def __init__(
        self,
        mechanism_id: str,
        name: str,
        category: str,
        description: str,
        hydrological_role: str,
        patterns: List[str],
        suitable_for: List[str],
        inputs: List[str],
        outputs: List[str],
        constraints: List[str],
        compatible_routing: List[str],
        related_models: List[str],
        implementation_template: str,
        params: Dict[str, List[float]]
    ):
        self.mechanism_id = mechanism_id
        self.name = name
        self.category = category
        self.description = description
        self.hydrological_role = hydrological_role
        self.patterns = patterns
        self.suitable_for = suitable_for
        self.inputs = inputs
        self.outputs = outputs
        self.constraints = constraints
        self.compatible_routing = compatible_routing
        self.related_models = related_models
        self.implementation_template = implementation_template
        self.params = params

    def to_dict(self) -> dict:
        return {
            "mechanism_id": self.mechanism_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "hydrological_role": self.hydrological_role,
            "patterns": self.patterns,
            "suitable_for": self.suitable_for,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "constraints": self.constraints,
            "compatible_routing": self.compatible_routing,
            "related_models": self.related_models,
            "implementation_template": self.implementation_template,
            "params": self.params
        }


class MechanismRegistry:
    """机制注册表 - 管理所有水文机制"""

    _instance = None
    _mechanisms: Dict[str, MechanismTemplate] = {}
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not MechanismRegistry._loaded:
            self._load_mechanisms()

    def _load_mechanisms(self):
        """从JSON文件加载机制定义"""
        if MechanismRegistry._loaded:
            return

        knowledge_dir = get_knowledge_dir()
        mechanisms_file = knowledge_dir / "mechanisms.json"

        if not mechanisms_file.exists():
            print(f"[Warning] mechanisms.json not found at {mechanisms_file}")
            return

        try:
            with open(mechanisms_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            all_mechanisms = []
            for category in ["runoff", "routing", "evap", "groundwater"]:
                if category in data:
                    all_mechanisms.extend(data[category])

            for mech_data in all_mechanisms:
                mech = MechanismTemplate(
                    mechanism_id=mech_data["mechanism_id"],
                    name=mech_data["name"],
                    category=mech_data["category"],
                    description=mech_data["description"],
                    hydrological_role=mech_data.get("hydrological_role", ""),
                    patterns=mech_data.get("patterns", []),
                    suitable_for=mech_data.get("suitable_for", []),
                    inputs=mech_data.get("inputs", []),
                    outputs=mech_data.get("outputs", []),
                    constraints=mech_data.get("constraints", []),
                    compatible_routing=mech_data.get("compatible_routing", []),
                    related_models=mech_data.get("related_models", []),
                    implementation_template=mech_data.get("implementation_template", ""),
                    params=mech_data.get("params", {})
                )
                self._mechanisms[mech.mechanism_id] = mech

            MechanismRegistry._loaded = True
            print(f"[Info] Loaded {len(self._mechanisms)} mechanisms")

        except Exception as e:
            print(f"[Error] Failed to load mechanisms: {e}")

    def get_mechanism(self, mechanism_id: str) -> Optional[MechanismTemplate]:
        """根据ID获取机制"""
        return self._mechanisms.get(mechanism_id)

    def get_mechanisms_by_category(self, category: str) -> List[MechanismTemplate]:
        """获取指定类别的所有机制"""
        return [m for m in self._mechanisms.values() if m.category == category]

    def get_mechanisms_by_pattern(self, pattern: str) -> List[MechanismTemplate]:
        """获取包含指定模式的机制"""
        return [m for m in self._mechanisms.values() if pattern in m.patterns]

    def get_mechanisms_for_basin(self, climate: str, area: float = None) -> List[MechanismTemplate]:
        """根据流域特征推荐机制"""
        suitable = []
        for mech in self._mechanisms.values():
            for cond in mech.suitable_for:
                if climate.lower() in cond.lower():
                    suitable.append(mech)
                    break
        return suitable

    def get_all_mechanisms(self) -> Dict[str, MechanismTemplate]:
        """获取所有机制"""
        return self._mechanisms.copy()


def get_registry() -> MechanismRegistry:
    """获取全局机制注册表实例"""
    return MechanismRegistry()