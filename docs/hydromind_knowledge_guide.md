# Hydromind 人工知识图谱构建指南

> 版本: v1.3.5
> 定位: Mechanism-aware Hydrological Model Synthesis Framework 的核心知识工程文档

---

## 目录

1. [知识图谱架构](#1-知识图谱架构)
2. [Mechanism 构建](#2-mechanism-构建)
3. [Pattern 构建](#3-pattern-构建)
4. [Constraint 构建](#4-constraint-构建)
5. [Model Decomposition 构建](#5-model-decomposition-构建)
6. [Implementation Template 构建](#6-implementation-template-构建)
7. [质量检查清单](#7-质量检查清单)
8. [附录：完整示例](#8-附录完整示例)

---

## 1. 知识图谱架构

### 1.1 整体结构

```
knowledge/
├── mechanisms.json          # [核心] 机制原子定义
├── patterns.json            # [核心] 跨模型结构模式抽象
├── constraints.json         # [核心] 物理约束 + 兼容性规则
├── model_decomposition.json # [参考] 经典模型拆解
└── templates/               # 机制级代码模板
    ├── runoff/              # 产流模板
    ├── routing/             # 汇流模板
    └── evap/                # 蒸散发模板
```

### 1.2 数据流

```
用户需求
    ↓
Mechanism JSON → KG-RAG 检索
    ↓
Pattern JSON → 结构模式匹配
    ↓
Constraint JSON → Reasoner 约束验证
    ↓
Template Python → Executer 代码生成
```

### 1.3 核心设计原则

| 原则 | 说明 |
|------|------|
| **原子性** | 每个 mechanism 是水文行为的不可分割单元 |
| **可组合** | 机制之间可自由组合，通过compatibility约束 |
| **物理可解释** | 每行模板代码对应明确的物理过程 |
| **小而精** | 控制在 10-20 个 mechanism，非大而全 |

---

## 2. Mechanism 构建

### 2.1 Mechanism 是什么

mechanism 是"水文行为的原子单元"，例如：

| 非机制（太大） | 机制（正确粒度） |
|---------------|----------------|
| XAJ 模型（完整模型） | saturation_excess（蓄满产流） |
| 降雨（过程） | threshold runoff（阈值触发产流） |
| 流域水量平衡（抽象概念） | linear_reservoir（线性水库汇流） |

**判定标准**：一个 mechanism 应该能被单独描述、单独实现代码、单独测试。

### 2.2 JSON Schema

```json
{
  "mechanism_id": "唯一标识符（snake_case）",
  "name": "中文名称",
  "category": "所属类别（runoff/routing/evap/groundwater）",

  "description": "物理机制的中文描述",

  "hydrological_role": "英文角色描述，用于LLM理解",

  "patterns": ["关联的结构模式ID列表"],

  "suitable_for": [
    "适用场景列表（气候区/流域类型/应用场景）"
  ],

  "inputs": [
    "物理输入变量列表（小写snake_case）"
  ],

  "outputs": [
    "物理输出变量列表"
  ],

  "constraints": [
    "必须遵守的约束ID列表"
  ],

  "compatible_routing": [
    "兼容的汇流机制ID列表"
  ],

  "related_models": [
    "包含此机制的经典模型列表"
  ],

  "implementation_template": "模板文件路径（相对于 templates/）",

  "params": {
    "参数名": [下限, 上限]
  }
}
```

### 2.3 新增 Mechanism 步骤

#### Step 1: 确认是否需要新建

对照 `knowledge/mechanisms.json` 现有列表，确认没有重复。

#### Step 2: 填写 JSON 字段

以下字段为**必须填写**：

- `mechanism_id` — 全局唯一，snake_case
- `category` — 必须属于已有 categories：runoff/routing/evap/groundwater
- `description` — 一句话物理描述
- `hydrological_role` — LLM prompt 会用到的英文角色描述
- `patterns` — 必须引用 `patterns.json` 中已定义的 pattern
- `inputs` / `outputs` — 命名与已有机制保持一致
- `constraints` — 必须引用 `constraints.json` 中已定义的约束
- `implementation_template` — 必须对应 `templates/` 中的实际文件

#### Step 3: 编写代码模板

详见[第 6 节](#6-implementation-template-构建)。

#### Step 4: 在 Model Decomposition 中验证

将新机制添加到 `model_decomposition.json` 中对应模型的 `mechanisms` 列表，确认分解合理。

### 2.4 命名规范

| 类别 | 命名模式 | 示例 |
|------|---------|------|
| 产流 | XXXX_runoff | saturation_excess |
| 汇流 | XXXX_routing（或直接名） | linear_reservoir |
| 蒸散发 | XXXX_et | layered_et |
| 地下水 | XXXX（直白描述） | recession |

### 2.5 推荐机制初始列表（已完成）

#### 产流（4个）

| mechanism_id | 核心物理 | 来源模型 |
|-------------|----------|---------|
| saturation_excess | 蓄满产流，阈值触发 | XAJ, TOPMODEL |
| infiltration_excess | 超渗产流，强度触发 | SCS-CN, Green-Ampt |
| soil_moisture_accounting | 土壤水分平衡，比例产流 | HBV, GR4J |
| storage_runoff | 简单蓄水产流 | Tank, 教育模型 |

#### 汇流（4个）

| mechanism_id | 核心物理 | 来源模型 |
|-------------|----------|---------|
| linear_reservoir | Q=kS, 指数退水 | XAJ, HBV, SAC |
| cascade | 多水库串联，延迟 | HBV-96, Tank, Nash |
| nonlinear_reservoir | Q=aS^b, 非线性 | VIC, 山洪 |
| nash_routing | Nash单位线 | 工程水文学 |

#### 蒸散发（2个）

| mechanism_id | 核心物理 | 来源模型 |
|-------------|----------|---------|
| layered_et | 分层蒸发，上层优先 | XAJ, Sacramento |
| soil_et | 土壤湿度比例蒸发 | HBV, GR4J |

---

## 3. Pattern 构建

### 3.1 Pattern 是什么

Pattern 是"跨模型共享的结构共性"，例如：

- **threshold** — XAJ 蓄满产流、SCS-CN 产流都用了阈值触发
- **cascade** — HBV 双水库、Tank 三水箱、Nash 串联都是串联结构
- **partition** — XAJ 自由水水源划分、HBV 土壤水分分配都是水量分配

### 3.2 为什么需要 Pattern

| 没有 Pattern | 有 Pattern |
|-------------|-----------|
| Planner 只能选"XAJ"完整模型 | Planner 能抽象出"threshold + reservoir"设计新模型 |
| 知识碎片化 | 发现跨模型共性 |
| LLM 容易产生"伪创新" | 约束在已知结构框架内 |

### 3.3 JSON Schema

```json
{
  "pattern_id": "唯一标识符",
  "name": "中文名称",
  "description": "模式描述",
  "mechanisms": ["使用了此模式的机制ID列表"],
  "examples": ["跨模型的实现示例"],
  "code_principle": "可编译的伪代码模板"
}
```

### 3.4 新增 Pattern 步骤

1. 从已有/新增 mechanism 中抽象共性
2. 提取 `code_principle` 伪代码模板
3. 填写示例
4. 在 mechanisms.json 中对应机制的 `patterns` 字段中引用

### 3.5 现有 Pattern 列表

| pattern_id | 核心思想 | 涉及机制 |
|-----------|---------|---------|
| threshold | 状态超阈值触发 | saturation_excess, infiltration_excess |
| reservoir | 蓄泄函数关系 | linear_reservoir, nonlinear_reservoir |
| cascade | 顺序串联 | cascade, nash_routing |
| partition | 按比例/规则分配 | soil_moisture, layered_et |
| delay | 时间延迟 | nash_routing, cascade |
| exponential_decay | 指数衰减 | linear_reservoir, recession |
| ratio | 比例缩放 | soil_et, soil_moisture |

---

## 4. Constraint 构建

### 4.1 Constraint 是什么

Constraint 是"系统的物理底线"，分为三类：

| 类型 | 作用 | enforcement |
|------|------|------------|
| physics | 物理定律（水量守恒、非负） | hard（不可违反） |
| structural | 结构要求（汇流需要产流输入） | hard/warning |
| compatibility | 机制兼容性 | hard/warning |

### 4.2 JSON Schema

```json
{
  "constraint_id": "唯一标识符",
  "type": "physics / structural / compatibility",
  "description": "约束描述",
  "rule": "约束规则文本",
  "enforcement": "hard（强制） / warning（警告）",
  "check_code": "可执行的约束检查代码（可选）",
  "applies_when": "适用条件（可选）"
}
```

### 4.3 新增 Constraint 步骤

1. 确定约束类型
2. 填写 rule 文本（清晰、无歧义）
3. 设置 enforcement 级别
4. 如果是**compatibility**规则，同时更新 `compatibility_rules` 列表

### 4.4 兼容性规则 Schema

```json
{
  "from": "产流机制ID",
  "to": "汇流机制ID",
  "compatible": true / false,
  "reason": "兼容/不兼容的原因"
}
```

### 4.5 约束覆盖检查清单

新增 mechanism 后，检查以下约束是否覆盖：

- [ ] 水量守恒约束
- [ ] 径流/储水非负约束
- [ ] 蒸散发受限制约束
- [ ] 结构依赖约束（汇流需要产流输入）
- [ ] 与现有机制的兼容性规则
- [ ] 气候适用性规则

---

## 5. Model Decomposition 构建

### 5.1 作用

Model Decomposition 是"反向映射表"，记录经典水文模型由哪些机制构成。作用：

1. **验证** — 检查机制分解是否合理
2. **溯源** — 回答"这个机制组合源于哪个经典模型"
3. **教学** — 展示模型的机制化理解

### 5.2 JSON Schema

```json
{
  "model_id": "模型标识符",
  "name": "模型全称",
  "description": "简介",
  "mechanisms": ["构成该模型的机制ID列表"],
  "patterns": ["使用的模式列表"],
  "structure": {
    "模块名": {
      "mechanism": "对应机制ID",
      "description": "该模块的详细描述"
    }
  },
  "suitable_for": ["适用场景"],
  "typical_params": {
    "参数名": [下限, 上限]
  }
}
```

### 5.3 新增模型分解步骤

1. 研究目标模型的结构
2. 将其拆解为 mechanism 原子的组合
3. 在 mechanisms.json 确保每个机制已定义
4. 在 each 模块中标注对应的 mechanism
5. 写出典型参数范围

### 5.4 分解示例

```
XAJ = [saturation_excess, layered_et, linear_reservoir, partition, threshold]
HBV = [soil_moisture_accounting, cascade, soil_et, ratio]
Tank = [storage_runoff, cascade, linear_reservoir, reservoir]
SCS-CN = [infiltration_excess, threshold, storage]
GR4J = [soil_et, storage_runoff, cascade, ratio]
```

---

## 6. Implementation Template 构建

### 6.1 原则

| 原则 | 说明 |
|------|------|
| **单一职责** | 一个模板只实现一个 mechanism |
| **纯函数** | 无副作用，输入→输出 |
| **物理注释** | 中文，标注每步的物理含义 |
| **约束内置** | 代码内实现硬约束（如 max(runoff, 0)） |

### 6.2 Python Template 规范

```python
"""
机制名称机制模板

物理描述
来源模型: 相关模型列表
"""

def mechanism_id(
    input_var1: np.ndarray,
    input_var2: float,
    params: dict
) -> Tuple[float, ...]:
    """
    一行中文描述
    
    参数:
        input_var1: 输入描述 (单位)
        input_var2: 输入描述 (单位)
        params: 参数字典 {"param_name": 默认值}
    
    返回:
        output1: 输出描述 (单位)
        output2: 输出描述 (单位)
    """
    # 参数读取
    param1 = params.get("param1", default_value)
    
    # 物理计算步骤（中文注释）
    storage_change = inflow - outflow
    
    # 物理约束（硬约束）
    outflow = max(outflow, 0)
    storage = max(storage, 0)
    
    return outflow, storage
```

### 6.3 添加新 Template 步骤

1. 在 `knowledge/templates/<category>/` 下创建 `.py` 文件
2. 实现纯函数
3. 在 `mechanisms.json` 中将 `implementation_template` 指向新文件
4. 添加物理约束（nonnegative guard, mass balance check）

### 6.4 现有模板检查

| 文件 | 对应的 mechanism_id | 状态 |
|------|-------------------|------|
| `runoff/saturation_excess.py` | saturation_excess | 已完成 |
| `runoff/infiltration_excess.py` | infiltration_excess | 已完成 |
| `runoff/soil_moisture.py` | soil_moisture_accounting | 已完成 |
| `routing/linear_reservoir.py` | linear_reservoir | 已完成 |
| `routing/cascade.py` | cascade | 已完成 |
| `evap/layered_et.py` | layered_et | 已完成 |
| `evap/soil_et.py` | soil_et | 已完成 |

---

## 7. 质量检查清单

### 7.1 新增 Mechanism 检查

- [ ] mechanism_id 是否全局唯一？
- [ ] category 是否在现有列表中？
- [ ] patterns 是否引用已定义的 pattern？
- [ ] constraints 是否引用已定义的 constraint？
- [ ] compatible_routing 是否符合物理规律？
- [ ] implementation_template 文件是否存在？
- [ ] params 是否包含合理的上下界？

### 7.2 新增 Pattern 检查

- [ ] 是否至少在 2 个 mechanism 中出现？
- [ ] code_principle 是否可编译？
- [ ] 示例是否跨模型？

### 7.3 新增 Constraint 检查

- [ ] type 是否正确分类？
- [ ] enforcement 是否合理（hard/warning）？
- [ ] 是否被至少 1 个 mechanism 引用？
- [ ] compatibility 规则是否双向一致？

### 7.4 新增 Template 检查

- [ ] 函数名与 mechanism_id 一致？
- [ ] 输入参数与 mechanisms.json 中 inputs 匹配？
- [ ] 返回值与 outputs 匹配？
- [ ] 包含 nonnegative guard？
- [ ] 包含水量守恒？
- [ ] 参数读取使用 params.get() 带默认值？

### 7.5 全局一致性检查

```python
# 检查机制引用一致性
mechanisms.json 中所有 mechanism:
    - patterns 必须存在于 patterns.json
    - constraints 必须存在于 constraints.json
    - implementation_template 对应的文件必须存在
    - compatible_routing 引用的 routing mechanism 必须存在

patterns.json 中所有 pattern:
    - 至少被 1 个 mechanism 引用

constraints.json 中所有 constraint:
    - 至少被 1 个 mechanism 引用
```

---

## 8. 附录：完整示例

### 8.1 新增一个机制的完整流程

假设需要新增 `interflow`（壤中流产流机制）：

#### Step 1: mechanisms.json

```json
{
  "mechanism_id": "interflow",
  "name": "壤中流",
  "category": "runoff",
  "description": "水分在土壤非饱和层内水平运动形成的径流",
  "hydrological_role": "Generate lateral subsurface flow within soil profile",
  "patterns": ["partition", "reservoir"],
  "suitable_for": ["humid", "sloping", "soil_deep"],
  "inputs": ["soil_moisture_excess", "slope", "soil_depth"],
  "outputs": ["interflow", "remaining_water"],
  "constraints": [
    "mass_conservation",
    "interflow >= 0"
  ],
  "compatible_routing": ["linear_reservoir", "cascade"],
  "related_models": ["xaj", "sacramento", "dhsvm"],
  "implementation_template": "runoff/interflow.py",
  "params": {
    "ki": [0.01, 0.3],
    "slope_factor": [0.5, 2.0]
  }
}
```

#### Step 2: constraints.json 添加约束

```json
{
  "constraint_id": "interflow_limited_by_moisture",
  "type": "physics",
  "description": "壤中流不能超过当前土壤过剩水量",
  "rule": "interflow <= soil_moisture_excess",
  "enforcement": "hard"
}
```

#### Step 3: templates/runoff/interflow.py

```python
"""
壤中流产流机制模板

水分在土壤非饱和层内侧向运动形成壤中流
来源模型: XAJ, Sacramento, DHSVM
"""

def interflow(soil_moisture_excess, slope, params):
    """
    壤中流计算
    
    参数:
        soil_moisture_excess: 土壤过剩水量 (mm)
        slope: 坡度 (m/km)
        params: 参数字典 {"ki": 侧向出流系数, "slope_factor": 坡度系数}
    
    返回:
        interflow: 壤中流量 (mm/day)
        remaining: 剩余水量
    """
    ki = params.get("ki", 0.1)
    slope_factor = params.get("slope_factor", 1.0)
    
    # 坡度修正出流系数
    adjusted_ki = ki * (1 + slope_factor * slope / 100)
    
    # 侧向出流
    interflow = adjusted_ki * soil_moisture_excess
    interflow = max(interflow, 0)
    
    # 剩余水量
    remaining = max(soil_moisture_excess - interflow, 0)
    
    return interflow, remaining
```

#### Step 4: model_decomposition.json 更新

```json
{
  "model_id": "xaj",
  "mechanisms": [
    "saturation_excess",
    "layered_et",
    "interflow",
    "linear_reservoir"
  ]
}
```

### 8.2 验证运行

```bash
# 启动 Streamlit 后选择机制模式
# 检查 Planner 是否能推荐 interflow
# 检查 Reasoner 兼容性检查是否通过
# 检查 Executer 是否能生成含壤中流的代码
```

---

> 本文档是 Hydromind v1.3.5 的核心知识工程指南。
> 维护者需要确保 knowledge/ 目录下 JSON 与 Python 文件的一致性。
> 每次修改后建议运行 `python build_vector_db.py` 重建向量索引。
