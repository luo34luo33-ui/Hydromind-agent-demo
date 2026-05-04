# Hydromind 更新说明

## v1.3.1_develop (最新版本)

**发布日期**: 2025-05-04

### 核心升级：模块化 Code Template 体系

从"完整模型"升级为"产流/汇流模块组合"架构，实现乐高式构建。

#### Phase 1: 模块接口设计 (modules.py)
- 定义 `RunoffModule` 和 `RoutingModule` 抽象基类
- **防雷设计**: `state_init_mapping` 状态初始化映射
- **防雷设计**: 参数后缀约束（`_runoff` / `_routing`）

```python
class RunoffModule(ABC):
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        """例如: {"soil_storage": "params.get('S0_runoff', 0.0)"}"""
        pass
    
    @property
    def params(self) -> Dict[str, List[float]]:
        """参数必须以 _runoff 结尾"""
        pass
```

#### Phase 2: 产流模块实现 (runoff_modules.py)

| 模块 ID | 名称 | 参数后缀 | state_keys |
|--------|------|----------|------------|
| `scs_runoff` | SCS-CN 产流 | `CN_runoff`, `Smax_runoff` | soil_storage |
| `xaj_runoff` | 新安江蓄满产流 | `WM_runoff`, `B_runoff` | W, S |
| `tank_runoff` | Tank 产流 | `k1_runoff`, `k2_runoff` | S1, S2, S3 |
| `hbv_runoff` | HBV 土壤水分 | `FC_runoff`, `BETA_runoff` | soil |
| `simple_runoff` | 简单产流 | `k_runoff` | storage |

#### Phase 3: 汇流模块实现 (routing_modules.py)

| 模块 ID | 名称 | 参数后缀 | state_keys |
|--------|------|----------|------------|
| `linear_routing` | 线性水库 | `k_routing`, `S0_routing` | routing_storage |
| `nonlinear_routing` | 非线性水库 | `k_routing`, `beta_routing` | routing_storage |
| `nash_routing` | Nash 单位线 | `k_routing`, `n_routing` | nash_1, nash_2, nash_3 |
| `direct_routing` | 直接汇流 | (无) | (无) |

#### Phase 4: 模型组合器 (composer.py)

**防雷设计实现**:

1. **参数命名空间隔离**:
```python
# 产流参数
params = {"CN_runoff": [30, 95], "Smax_runoff": [0, 500]}
# 汇流参数  
params = {"k_routing": [0.05, 0.8], "S0_routing": [0, 200]}
# 合并后无冲突
```

2. **状态初始化映射**:
```python
# 使用模块的 state_init_mapping
initial_state = {
    "soil_storage": "params.get('S0_runoff', 0.0)",
    "routing_storage": "params.get('S0_routing', 50.0)"
}
```

3. **只提取计算核心**:
```python
# 只提取 compute/route 方法，丢弃类的外壳
def _get_module_method_code(self, module):
    if hasattr(module, 'compute'):
        return inspect.getsource(module.compute)
```

#### Phase 5-6: Planner + main.py 集成

**Planner Schema 更新**:
```python
class ModelingPlan(BaseModel):
    runoff_module_id: Literal[
        "scs_runoff", "xaj_runoff", "tank_runoff", "hbv_runoff", "simple_runoff"
    ]
    routing_module_id: Literal[
        "linear_routing", "nonlinear_routing", "nash_routing", "direct_routing"
    ]
```

**main.py 双模式支持**:
- 新模式：`ModelComposer.compose(runoff_id, routing_id)`
- 兼容模式：旧版模板系统

#### 数据流

```
Planner:
  - runoff_module_id: "scs_runoff"
  - routing_module_id: "linear_routing"
         ↓
ModelComposer.compose():
  1. 获取模块代码 (inspect.getsource 只提取方法)
  2. 合并参数 (自动加 _runoff / _routing 后缀)
  3. 生成状态初始化 (使用 state_init_mapping)
  4. 生成完整代码 (Blueprint + 模块代码)
         ↓
simulate_runoff(precip, pet, params)
         ↓
SCE-UA 率定 (使用带后缀的参数)
```

#### 新增文件

| 文件 | 说明 |
|------|------|
| `src/templates/modules.py` | 接口定义 + 防雷设计 |
| `src/templates/runoff_modules.py` | 5个产流模块 |
| `src/templates/routing_modules.py` | 4个汇流模块 |
| `src/templates/composer.py` | 组合器 + 蓝图 |

#### 收益

- **乐高式组合**: 产流模块 × 汇流模块 = 完整模型
- **状态隔离**: `state` 字典管理，避免变量混乱
- **零参数冲突**: 自动后缀隔离
- **水量守恒**: 模块内部守恒，组合后依然守恒

---

## v1.3_develop

**发布日期**: 2025-05-04

### 核心升级：Prompt 修正 + Python Registry + 精确 ID 寻址

#### Phase 3: Executer Prompt 修正
- 修改 Prompt 措辞，添加"核心物理机制约束"
- 明确告诉 LLM **必须绝对保留水量平衡（Mass Balance）机制**
- 添加"接口适配约束"：仅修改参数读取方式和输入输出接口
- 移除危险的暗示"不要完全照抄"

**修改后的 Prompt**:
```
参考基准代码模板：
{code_template}

【开发指令】：
1. 核心物理机制约束：必须绝对保留上述模板中的产流/汇流核心逻辑和水量平衡（Mass Balance）机制，切勿随意修改物理运算步骤。
2. 接口适配约束：请在此模板基础上，根据 Planner 提供的参数建议（{param_list_str}），仅修改参数读取方式（如 params.get(...)）和输入输出接口。
```

#### Phase 2: Python Registry (使用 inspect)
- 新建 `src/templates/registry.py`
- 使用 `inspect.getsource()` 动态获取函数源码
- 包含 5 个模板函数：`_linear_reservoir_simulate`, `_tank_model_simulate`, `_scs_cn_simulate`, `_xaj_simulate`, `_hbv_simulate`
- **删除** 脆弱的 Markdown 正则解析

**模板注册表**:
```python
class ModelTemplate:
    id: str
    name: str
    keywords: List[str]
    func: callable  # 传入真实函数对象
    params: Dict[str, List[float]]

TEMPLATE_REGISTRY = {
    "linear_reservoir": ModelTemplate(id="linear_reservoir", ...),
    "tank_model": ModelTemplate(id="tank_model", ...),
    "scs_cn": ModelTemplate(id="scs_cn", ...),
    "xaj": ModelTemplate(id="xaj", ...),
    "hbv": ModelTemplate(id="hbv", ...),
}
```

#### Phase 1: 精确 ID 寻址 (O(1) 匹配)
- Planner 的 `ModelingPlan` 中 `template_keywords` → `template_ids`
- 使用 `Literal` 限制可选值，锁死模板 ID
- `CodeTemplateRAG.get_templates_by_ids()` 实现 O(1) 寻址

**修改后的 Schema**:
```python
class ModelingPlan(BaseModel):
    # ...
    template_ids: List[Literal["linear_reservoir", "tank_model", "scs_cn", "xaj", "hbv"]] = Field(
        description="建议 Executer 提取的代码模板 ID"
    )
```

#### 数据流对比

| 版本 | 检索方式 | 匹配逻辑 |
|------|----------|----------|
| v1.2 | `retrieve_by_keywords()` | 关键词匹配（概率游戏） |
| v1.3 | `get_templates_by_ids()` | 精确 ID 寻址（O(1)） |

#### 新增文件
| 文件 | 说明 |
|------|------|
| `src/templates/__init__.py` | 模板包初始化 |
| `src/templates/registry.py` | Python Registry 注册表 |

#### 修改文件
| 文件 | 改动 |
|------|------|
| `src/agents/executer.py` | Prompt 措辞修正，添加物理约束 |
| `src/agents/planner.py` | template_keywords → template_ids |
| `src/utils/rag_engine.py` | 从 Registry 加载，添加 get_templates_by_ids() |
| `src/main.py` | 调用 get_templates_by_ids() |

#### 收益
- **零正则**：彻底告别 Markdown 解析异常
- **O(1) 匹配**：100% 命中率，无概率波动
- **物理锁定**：减少 LLM 修改物理公式的风险
- **IDE 支持**：模板代码享受语法高亮和格式化

---

## v1.2_develop

**发布日期**: 2025-05-04

### 核心升级：彻底消灭正则提取，实现全链路 JSON 契约

#### Phase 1: Planner 结构化输出
- 新增 `ModelingPlan` Pydantic 模型
- 产流/汇流分离：`runoff_type` + `flow_routing`
- 参数建议穿透：`param_suggestions: Dict[str, List[float]]`
- 模板关键词：`template_keywords` 自动触发代码模板检索
- 保留 `try-except` 回退机制

**Schema 定义**:
```python
class ModelingPlan(BaseModel):
    reasoning: str
    runoff_type: Literal["SCS-CN", "新安江", "HBV", "Tank Model"]
    flow_routing: Literal["线性水库", "非线性水库", "Nash瞬时单位线", "马斯京根"]
    response_type: Literal["快速响应", "中等响应", "慢速响应"]
    param_suggestions: Dict[str, List[float]]
    description: str
    template_keywords: List[str]
```

#### Phase 2: Executer 结构化输出
- 扩展 `HydroModelCode`，新增 `parameters_config` 字段
- 参数约束注入 Prompt：`param_constraints` 来自 Planner
- 方法返回值改为 dict：`{"code": str, "parameters_config": dict, "reasoning": str}`

**Schema 定义**:
```python
class HydroModelCode(BaseModel):
    reasoning: str
    simulate_function: str
    parameters_config: Dict[str, List[float]]  # 直接给 SCE-UA
```

#### Phase 3: main.py 集成
- Planner 启用 `use_structured=True`（针对 gpt-4o/gpt-4o-mini）
- 提取 `param_constraints` 传给 Executer
- 提取 `parameters_config` 直接传给 SCE-UA
- **移除** `extract_params_from_code()` 正则调用
- 新增 `CodeTemplateRAG.retrieve_by_keywords()` 方法

#### 数据流（v1.2）

```
Planner (ModelingPlan)
    ├── runoff_type: SCS-CN/新安江/HBV/Tank
    ├── flow_routing: 线性水库/非线性水库/Nash/马斯京根
    ├── param_suggestions: {"k": [0.05, 0.8], ...}
    └── template_keywords: ["Tank", "水箱", ...]
           ↓
CodeTemplateRAG.retrieve_by_keywords()
           ↓
Executer (HydroModelCode)
    ├── simulate_function: "def simulate_runoff..."
    └── parameters_config: {"k": [0.05, 0.8], ...}  ← 直接给 SCE-UA
           ↓
SCE-UA (无需正则提取，直接使用 parameters_config)
```

#### 新增文件
| 文件 | 说明 |
|------|------|
| `docs/hydro_code_templates.md` | 5个标准水文模型代码模板 |
| `src/agents/graph.py` | LangGraph 状态机 |

#### 修改文件
| 文件 | 改动 |
|------|------|
| `src/agents/planner.py` | +64 行：结构化输出 |
| `src/agents/executer.py` | +111 行：参数穿透 |
| `src/main.py` | +151 行：集成新流程 |
| `src/utils/rag_engine.py` | +136 行：CodeTemplateRAG |
| `requirements.txt` | +langgraph, +numba |

#### 收益
- **零正则**：彻底告别 AST/正则解析
- **JSON 契约**：模块间纯 JSON 接口
- **参数可溯**：从 Planner 一路穿透到 SCE-UA
- **鲁棒性**：LLM 输出格式变化不影响下游

---

## v1.1_develop

**发布日期**: 2025-05-04

### 核心升级：代码模板检索 + LangGraph 状态机

#### 1. 代码模板检索
- 新建 `docs/hydro_code_templates.md`
- 包含 5 个标准水文模型模板：线性水库、Tank Model、SCS-CN、新安江、HBV
- 每个模板包含：适用场景、参数范围、完整代码

#### 2. 扩展 RAG 引擎
- 新增 `CodeTemplateRAG` 类
- 支持 `retrieve_by_plan(plan_text)` 自动匹配
- 关键词映射：慢响应→线性水库，湿润区→新安江，等

#### 3. LangGraph 状态机
- 新建 `src/agents/graph.py`
- 实现 Actor-Critic 循环
- 节点：`generate` → `validate`
- 条件边：失败自动重试（最多3次）

#### 4. Executer 增强
- 支持代码模板注入到 Prompt
- 支持 `use_structured=True` 结构化输出
- 新增 `HydroModelCode` Pydantic 模型

#### 5. main.py 集成
- 集成代码模板检索流程
- 调用流程：Planner → 检索模板 → Executer

---

## 对比总结

| 版本 | 核心特性 | 状态 |
|------|----------|------|
| v1.0 | 基础 Agent 流程 | 已发布 |
| v1.1 | 代码模板检索 + LangGraph | 已发布 |
| v1.2 | Planner/Executer 双结构化 + 零正则 | 已发布 |
| v1.3 | Prompt 修正 + Python Registry + 精确 ID | 已发布 |
| v1.3.1 | 模块化 Code Template (产流/汇流模块组合) | 开发中 |

---

## 依赖更新

```diff
 requirements.txt
+langgraph>=1.0.0,<2.0.0
+numba>=0.58.0
```

---

## 后续规划

- [ ] v1.3 集成 LangGraph 状态机到 main.py
- [ ] 添加更多代码模板（SWAT、水动力模型）
- [ ] 支持自定义参数约束
- [ ] 添加多模型对比功能
- [ ] 模板单元测试验证