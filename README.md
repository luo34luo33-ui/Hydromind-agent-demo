# Hydromind - 流域水文模型开发智能体

<p align="center">
  <img src="https://img.shields.io/badge/Version-v1.3_develop-blue" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/AI-LangChain/LangGraph-orange" alt="AI Framework">
</p>

Hydromind 是一款基于大语言模型（LLM）的流域水文模型自动开发系统。通过多智能体协作，自动完成水文模型设计、代码生成、参数率定和误差校正。

---

## 功能特性

### 🤖 多智能体协作
- **Planner Agent**: 分析流域特征，设计产流/汇流方案
- **Executer Agent**: 生成水文模型模拟代码
- **Validator Agent**: 验证代码语法和执行正确性
- **SCE-UA Optimizer**: 全局参数率定
- **ML Corrector**: XGBoost 误差校正

### 📊 核心能力
- 自动识别流域特征（面积、坡度、渗透性、气候类型）
- 智能推荐水文模型（线性水库、Tank Model、SCS-CN、新安江、HBV）
- 全局优化参数率定（SCE-UA 算法）
- 误差校正提升模拟精度（XGBoost）
- 一键导出完整模型包

### 🛠️ 技术架构
- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **框架**: LangChain + LangGraph
- **前端**: Streamlit
- **优化算法**: SCE-UA 全局优化
- **机器学习**: XGBoost 残差校正

---

## 版本演进

| 版本 | 核心特性 | 日期 |
|------|----------|------|
| v1.0 | 基础 Agent 流程 | 2025-04 |
| v1.1 | 代码模板检索 + LangGraph 重试 | 2025-05-04 |
| v1.2 | Planner/Executer 双结构化 + 零正则提取 | 2025-05-04 |
| v1.3 | Prompt 修正 + Python Registry + 精确 ID 寻址 | 2025-05-04 |

---

## 快速开始

### 环境要求
- Python 3.9+
- OpenAI API Key

### 安装

```bash
# 克隆项目
git clone https://github.com/luo34luo33-ui/Hydromind-agent-demo.git
cd Hydromind-agent-demo

# 创建虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
cd src
streamlit run main.py
```

### 使用流程

1. 在侧边栏输入 OpenAI API Key
2. 选择 LLM 模型（推荐 gpt-4o）
3. 选择目标流域
4. 可选：输入建模需求（如"使用 Tank Model"）
5. 点击"开始智能建模"
6. 查看模拟结果，导出模型

---

## 项目结构

```
Hydromind-agent-demo/
├── src/
│   ├── agents/           # 多智能体模块
│   │   ├── planner.py    # Planner Agent
│   │   ├── executer.py   # Executer Agent
│   │   ├── validator.py  # Validator Agent
│   │   └── graph.py      # LangGraph 状态机
│   ├── templates/        # 代码模板注册表
│   │   └── registry.py   # Python Registry
│   ├── simulation/       # 优化和校正模块
│   │   ├── sceua.py      # SCE-UA 率定
│   │   └── ml_correction.py  # XGBoost 校正
│   ├── utils/            # 工具模块
│   │   ├── rag_engine.py    # RAG 引擎
│   │   └── data_loader.py   # 数据加载
│   └── main.py           # Streamlit 主界面
├── docs/                 # 文档
│   ├── hydro_knowledge.md
│   └── hydro_code_templates.md
├── CHANGELOG.md          # 更新日志
└── requirements.txt      # 依赖列表
```

---

## 技术亮点

### 1. 结构化输出
Planner 和 Executer 使用 Pydantic 进行结构化输出，模块间通过 JSON 契约通信，零正则提取。

### 2. Python Registry
代码模板存储在 `src/templates/registry.py`，使用 `inspect.getsource()` 动态获取源码，告别脆弱的 Markdown 解析。

### 3. 精确 ID 寻址
Planner 输出确定的模板 ID（如 `["tank_model"]`），实现 O(1) 匹配，100% 命中率。

### 4. 物理约束
Executer 的 Prompt 包含"核心物理机制约束"，锁死水量平衡公式，防止 LLM 随意修改物理逻辑。

---

## 数据要求

输入数据格式（CSV）:
```csv
date,precip,pet,q_obs
2020-01-01,5.2,2.1,1.3
2020-01-02,0.0,2.3,1.1
...
```

| 字段 | 说明 | 单位 |
|------|------|------|
| date | 日期 | YYYY-MM-DD |
| precip | 日降雨量 | mm/day |
| pet | 日潜在蒸散发 | mm/day |
| q_obs | 日观测径流 | mm/day |

---

## 性能指标

- **NSE**: Nash-Sutcliffe 效率系数
- **KGE**: Kling-Gupta 效率系数
- 支持率定期和验证期双评估

---

## 依赖

```
numpy>=1.24.0
pandas>=2.0.0
xgboost>=2.0.0
plotly>=5.0.0
streamlit>=1.28.0
chromadb>=0.4.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-text-splitters>=0.0.0
langgraph>=1.0.0,<2.0.0
numba>=0.58.0
```

---

## 后续规划

- [ ] v1.3 集成 LangGraph 状态机到 main.py
- [ ] 添加更多代码模板（SWAT、水动力模型）
- [ ] 支持自定义参数约束
- [ ] 添加多模型对比功能
- [ ] 模板单元测试验证

---

## 许可证

MIT License

---

## 贡献者

Hydromind 由大语言模型辅助开发，设计理念来自水文模型与 Agent 技术的融合实践。