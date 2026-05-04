# Hydromind 更新说明

## v1.2_develop (最新版本)

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
| v1.0 | 基础 Agent 流程 (Planner → Executer → Validator → SCE-UA) | 已发布 |
| v1.1 | 代码模板检索 + LangGraph 重试机制 | 已发布 |
| v1.2 | Planner/Executer 双结构化 + 零正则提取 | 开发中 |

---

## 依赖更新

```diff
 requirements.txt
+langgraph>=1.0.0,<2.0.0
+numba>=0.58.0
```

---

## 后续规划

- [ ] v1.2 集成 LangGraph 状态机到 main.py
- [ ] 添加更多代码模板（SWAT、水动力模型）
- [ ] 支持自定义参数约束
- [ ] 添加多模型对比功能