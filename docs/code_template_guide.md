# Code Template 写作指南

本文档说明如何在 `docs/hydro_code_templates.md` 中编写水文模型代码模板。

> **注意**：v1.3 已迁移到 Python Registry（`src/templates/registry.py`），Markdown 文件作为文档参考保留。

---

## 模板结构

每个模板必须包含以下部分：

```markdown
## 模型名称 (模型英文名)

### 适用场景
- 场景描述1
- 场景描述2

### 参数范围
| 参数 | 下界 | 上界 | 单位 | 说明 |
|------|------|------|------|------|
| k    | 0.05 | 0.80 | 无量纲 | 出流系数 |

### 代码模板
```python
def simulate_runoff(precip, pet, params):
    # 代码内容
    return q
```

### 注意事项
- 注意事项1
```

---

## 规则详解

### 1. 模型名称格式

```
## 模型名称 (模型英文名)
```

示例：
```markdown
## 线性水库模型 (Linear Reservoir)
## Tank Model - 三水箱 (Three-Tank Model)
```

### 2. 适用场景

使用简短的 bullet 列表，描述适合使用该模型的流域特征：

```markdown
### 适用场景
- 慢速响应流域
- 低坡度、土壤渗透性低
- 干旱或半干旱气候
- 中大型流域面积
```

### 3. 参数范围表

使用 Markdown 表格，格式必须严格：

| 参数 | 下界 | 上界 | 单位 | 说明 |
|------|------|------|------|------|
| k    | 0.05 | 0.80 | 无量纲 | 出流系数，越大出流越快 |
| S0   | 0    | 200  | mm    | 初始储水量 |

**规则**：
- 第一列：参数名（小写字母+数字）
- 第二列：下界（数字）
- 第三列：上界（数字）
- 第四列：单位（中文或无量纲）
- 第五列：说明（简短描述）

### 4. 代码模板

```python
def simulate_runoff(precip, pet, params):
    """
    函数说明（可选）

    参数:
        precip: numpy 数组，日降雨量 (mm/day)
        pet: numpy 数组，日潜在蒸散发 (mm/day)
        params: 字典，模型参数

    返回:
        q: numpy 数组，日径流序列 (mm/day)
    """
    # 代码内容
    return q
```

**规则**：
- 必须使用 `def simulate_runoff(precip, pet, params):` 签名
- 不要写 import 语句（假设 `np` 已可用）
- 必须包含完整的水量平衡
- 必须包含 `params.get("参数名", 默认值)`
- 注释使用中文
- 长度无限制

### 5. 注意事项

简要说明使用该模型时的关键点：

```markdown
### 注意事项
- 无降雨时仍有基流（因为 S > 0）
- 储水衰减呈指数形式
- 适用于单一蓄水层的简单流域
```

---

## 完整示例

### 示例：线性水库模型

```markdown
## 线性水库模型 (Linear Reservoir)

### 适用场景
- 慢速响应流域
- 低坡度、土壤渗透性低
- 干旱或半干旱气候
- 中大型流域面积

### 参数范围
| 参数 | 下界 | 上界 | 单位 | 说明 |
|------|------|------|------|------|
| k    | 0.05 | 0.80 | 无量纲 | 出流系数，越大出流越快 |
| S0   | 0    | 200  | mm    | 初始储水量 |

### 代码模板
```python
def simulate_runoff(precip, pet, params):
    """
    线性水库模型 - 单水箱水量平衡
    
    水量平衡: dS/dt = P - ET - Q
    汇流公式: Q = k * S
    
    参数:
        precip: 日降雨量序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        params: 参数字典 {"k": 出流系数, "S0": 初始储水量}
    
    返回:
        q: 日径流序列 (mm/day)
    """
    k = params.get("k", 0.3)
    S0 = params.get("S0", 50.0)
    n = len(precip)
    
    q = np.zeros(n)
    S = S0
    
    for t in range(n):
        S = S + precip[t] - 0.5 * pet[t]
        if S < 0:
            S = 0
        q[t] = k * S
        S = S - q[t]
    
    return q
```

### 注意事项
- 无降雨时仍有基流（因为 S > 0）
- 储水衰减呈指数形式
- 适用于单一蓄水层的简单流域
```

---

## 添加新模板

1. **选择唯一 ID**：在 `registry.py` 中添加 `TEMPLATE_REGISTRY` 条目
2. **编写函数**：实现 `_xxx_simulate(precip, pet, params)` 函数
3. **定义参数**：在 `params` 中定义参数范围
4. **添加关键词**：在 `keywords` 中添加检索关键词

### registry.py 示例

```python
def _my_model_simulate(precip, pet, params):
    # 实现代码
    return q

TEMPLATE_REGISTRY = {
    "my_model": ModelTemplate(
        id="my_model",
        name="我的模型",
        keywords=["关键词1", "关键词2"],
        func=_my_model_simulate,
        params={"param1": [0.0, 1.0], "param2": [0, 100]},
        description="模型描述"
    ),
}
```

---

## 常见问题

### Q: 模板代码需要 import 吗？

**不需要**。Executer 会假设 `numpy as np` 已可用。

### Q: 可以使用第三方库吗？

**谨慎**。当前沙盒环境只预置了：
- `np` (numpy)
- `pd` (pandas)
- `jit` (numba.jit)

如需其他库，请在注意事项中说明。

### Q: 参数命名有什么规则？

- 使用小写字母和下划线：`k`, `S0`, `k1`, `CN`
- 避免与 Python 关键字冲突
- 保持与 SCE-UA 参数名一致

### Q: 如何测试模板？

1. 在 Python 中导入：
   ```python
   from src.templates.registry import get_template
   template = get_template("tank_model")
   print(template.code)
   ```

2. 执行模板代码：
   ```python
   import numpy as np
   exec(template.code)
   result = simulate_runoff(precip, pet, params)
   ```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/templates/registry.py` | Python Registry（主存储） |
| `docs/hydro_code_templates.md` | Markdown 文档（参考） |
| `src/utils/rag_engine.py` | 模板检索引擎 |