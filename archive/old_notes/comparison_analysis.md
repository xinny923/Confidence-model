# analyze_refined.py vs mdiscrete.py 详细对比分析

## 概述

这两个脚本都是基于 Chong et al. (2022) 论文的置信度模型实现，但 `mdiscrete.py` 是对原始模型的**扩展版本**，引入了第三种决策类型。

---

## 核心区别总结

### 1. **动作分类系统**

#### `analyze_refined.py` (原始模型)
- **二元分类**：只有两种动作
  - `accept = 1`：接受AI建议（`final_move == ai_suggestion`）
  - `accept = 0`：拒绝AI建议（`final_move != ai_suggestion`，即坚持原始选择）

```python
# analyze_refined.py 第248行
accept = int(aisugg == final_move)
act_values.append(accept)
```

#### `mdiscrete.py` (扩展模型)
- **三元分类**：三种动作类型
  - `action = 0`：接受AI（`final_move == ai_suggestion`）
  - `action = 1`：拒绝AI（`final_move == original_move`，坚持原始选择）
  - `action = 2`：修改（`final_move` 既不是AI建议也不是原始选择）

```python
# mdiscrete.py 第213-219行
if final_move == ai_suggestion:
    action = 0  # Accept AI
elif final_move == original_move:
    action = 1  # Reject AI (stick to original)
else:
    action = 2  # Modify (neither AI nor original)
```

**关键洞察**：`mdiscrete.py` 识别了第三种决策策略——"修改"，这代表参与者既不完全接受AI，也不坚持原始选择，而是做出了第三种选择。

---

### 2. **经验矩阵维度**

#### `analyze_refined.py` (原始模型)
- **4种经验类型**：`e_matrix.shape = (NUM_TRIALS, 4)`
  - `e1`: 接受AI → 正反馈
  - `e2`: 拒绝AI → 正反馈
  - `e3`: 接受AI → 负反馈
  - `e4`: 拒绝AI → 负反馈

```python
# analyze_refined.py 第237行
e_matrix = np.zeros((Config.NUM_TRIALS, 4), dtype=float)

# 第252-256行：经验矩阵构建
if not math.isnan(feedback2):
    sign = int(feedback2 / 5 * -1)  # -1 for positive, +1 for negative
    idx = (sign + 1) if accept else (sign + 2)
    if 0 <= idx < 4:
        e_matrix[t, idx] = 1.0
```

#### `mdiscrete.py` (扩展模型)
- **6种经验类型**：`e_matrix.shape = (NUM_TRIALS, 6)`
  - `e1`: 接受AI → 正反馈
  - `e2`: 拒绝AI → 正反馈
  - `e3`: 接受AI → 负反馈
  - `e4`: 拒绝AI → 负反馈
  - `e5`: **修改 → 正反馈** (新增)
  - `e6`: **修改 → 负反馈** (新增)

```python
# mdiscrete.py 第200行
e_matrix = np.zeros((Config.NUM_TRIALS, 6), dtype=float)  # Now 6 columns!

# 第224-236行：扩展的经验矩阵构建
if not math.isnan(feedback2):
    sign = int(feedback2 / 5 * -1)
    
    if action == 0:  # Accept AI
        idx = 0 if sign == -1 else 2  # e1 or e3
    elif action == 1:  # Reject (original)
        idx = 1 if sign == -1 else 3  # e2 or e4
    else:  # action == 2, Modify
        idx = 4 if sign == -1 else 5  # e5 or e6 (NEW!)
    
    if 0 <= idx < 6:
        e_matrix[t, idx] = 1.0
```

---

### 3. **模型参数数量**

#### `analyze_refined.py` (原始模型)
- **8个参数**：
  - `alpha_e`, `alpha_a`, `alpha_b` (3个学习率参数)
  - `omega1`, `omega2`, `omega3`, `omega4` (4个经验权重)
  - `gamma` (1个累积置信度衰减参数)

```python
# analyze_refined.py 第55-59行
PARAMETER_NAMES = [
    "alpha_e", "alpha_a", "alpha_b",
    "omega1", "omega2", "omega3", "omega4",
    "gamma"
]
```

#### `mdiscrete.py` (扩展模型)
- **10个参数**：
  - `alpha_e`, `alpha_a`, `alpha_b` (3个学习率参数)
  - `omega1`, `omega2`, `omega3`, `omega4`, **`omega5`, `omega6`** (6个经验权重)
  - `gamma` (1个累积置信度衰减参数)

```python
# mdiscrete.py 第44-48行
PARAMETER_NAMES = [
    "alpha_e", "alpha_a", "alpha_b",
    "omega1", "omega2", "omega3", "omega4", "omega5", "omega6",  # 6 omegas!
    "gamma"
]
```

---

### 4. **置信度动态更新方程**

#### `analyze_refined.py` (原始模型)
```python
# 第540-545行
experience = (
    omega1 * e_tensor[:, t, 0] +  # Accept → Positive
    omega2 * e_tensor[:, t, 1] +  # Reject → Positive
    omega3 * e_tensor[:, t, 2] +  # Accept → Negative
    omega4 * e_tensor[:, t, 3]    # Reject → Negative
)
```

#### `mdiscrete.py` (扩展模型)
```python
# 第417-424行
experience = (
    omega1 * e_tensor[:, t, 0] +  # Accept → Positive
    omega2 * e_tensor[:, t, 1] +  # Reject → Positive
    omega3 * e_tensor[:, t, 2] +  # Accept → Negative
    omega4 * e_tensor[:, t, 3] +  # Reject → Negative
    omega5 * e_tensor[:, t, 4] +  # Modify → Positive (NEW!)
    omega6 * e_tensor[:, t, 5]    # Modify → Negative (NEW!)
)
```

---

### 5. **数据记录结构**

#### `analyze_refined.py`
```python
@dataclass
class ParticipantData:
    e_matrix: np.ndarray  # (NUM_TRIALS, 4)
    act_series: np.ndarray  # Binary: 0 or 1
    # 没有 original_move_series
```

#### `mdiscrete.py`
```python
@dataclass
class ParticipantData:
    e_matrix: np.ndarray  # (NUM_TRIALS, 6) - now 6 experiences!
    act_series: np.ndarray  # Ternary: 0=accept, 1=reject, 2=modify
    original_move_series: np.ndarray  # 新增：记录原始选择
```

---

### 6. **逻辑回归分析**

#### `analyze_refined.py`
- 因变量：`accept` (二元：接受=1, 拒绝=0)
- 直接使用 `accept` 字段

```python
# 第557-576行
def prepare_logit_data(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    accept = matrices["act_matrix"][:, 1:]
    # accept 已经是 0/1 二元变量
```

#### `mdiscrete.py`
- 因变量：`accept` (二元：接受=1, 拒绝/修改=0)
- 需要将三元动作转换为二元：接受(0) → 1，拒绝(1)或修改(2) → 0

```python
# 第907-928行
def prepare_logit_data(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    accept = matrices["act_matrix"][:, 1:]
    # 转换为二元：accept (1) vs reject/modify (0)
    accept_binary = 1 if accept[idx, trial] == 0 else 0
```

---

### 7. **输出目录和文件命名**

#### `analyze_refined.py`
- 输出目录：`outputs/`
- 文件命名：`table1_model_params.csv`, `trial_data.csv`

#### `mdiscrete.py`
- 输出目录：`outputs_extended/`
- 文件命名：`table1_extended_model_params.csv`, `trial_data_extended.csv`

---

### 8. **新增分析功能**

#### `mdiscrete.py` 独有的功能：

1. **动作分布分析** (`analyze_action_distribution`)
   - 分析三种动作类型的分布
   - 按条件统计接受/拒绝/修改的比例

2. **动作分布可视化** (`plot_action_distribution`)
   - 绘制三种动作随时间的变化趋势

3. **按动作类型的置信度分析** (`plot_confidence_by_action`)
   - 分别分析接受/拒绝/修改时的AI置信度和自我置信度

```python
# mdiscrete.py 第806-832行
def analyze_action_distribution(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze distribution of 3 action types."""
    # 统计三种动作的分布
```

---

## 理论意义

### 原始模型 (analyze_refined.py)
- 基于论文的**标准实现**
- 假设决策是**二元选择**：要么接受AI，要么拒绝
- 符合经典的接受/拒绝决策框架

### 扩展模型 (mdiscrete.py)
- **理论扩展**：识别了第三种决策策略
- **"修改"动作的心理学意义**：
  - 可能代表**部分接受**AI建议
  - 可能代表**创造性整合**AI和人类知识
  - 可能代表**不确定性下的折中策略**
- **模型复杂度增加**：从8参数增加到10参数
- **可能提高模型拟合度**：如果"修改"动作确实存在且有意义

---

## 代码结构对比

| 特性 | analyze_refined.py | mdiscrete.py |
|------|-------------------|--------------|
| 动作分类 | 二元 (接受/拒绝) | 三元 (接受/拒绝/修改) |
| 经验类型 | 4种 | 6种 |
| 模型参数 | 8个 | 10个 |
| 经验矩阵维度 | (30, 4) | (30, 6) |
| 初始参数 | 来自论文目标值 | 扩展猜测值 |
| 输出目录 | `outputs/` | `outputs_extended/` |
| 额外分析 | 无 | 动作分布、置信度按动作分析 |

---

## 使用建议

### 使用 `analyze_refined.py` 当：
- 需要**复现论文原始结果**
- 关注**标准二元决策框架**
- 需要与**已发表论文结果对比**

### 使用 `mdiscrete.py` 当：
- 数据中存在**大量"修改"动作**（既不是AI建议也不是原始选择）
- 想要**探索更复杂的决策策略**
- 需要**更细粒度的经验分类**
- 研究**部分接受或创造性整合**行为

---

## 潜在研究问题

1. **"修改"动作的频率**：在数据中，"修改"动作占多大比例？
2. **模型拟合度比较**：扩展模型是否显著提高了拟合度（R²）？
3. **参数解释**：`omega5` 和 `omega6` 的值是否有心理学意义？
4. **决策模式**：不同性能水平的参与者是否使用"修改"策略的频率不同？

---

## 总结

`mdiscrete.py` 是对 `analyze_refined.py` 的**理论扩展**，主要创新在于：
1. 识别了第三种决策类型（修改）
2. 将经验矩阵从4维扩展到6维
3. 增加了2个模型参数（omega5, omega6）
4. 提供了更细粒度的行为分析

这种扩展可能有助于更好地理解人类-AI协作中的**复杂决策策略**，特别是当参与者不完全接受或拒绝AI建议时的行为模式。

