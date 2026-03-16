# DeepVIDv2 × Attention-Residuals 实验 Proposal

## 1. 目标

本项目拟在 **DeepVIDv2** 上引入 **Attention-Residual (AttnRes)** 机制，系统研究：

> 在自监督 low-photon voltage imaging 去噪任务中，沿网络深度的可学习特征聚合，是否优于标准固定 residual；以及其最优注入强度、注入位置与复杂度收益比。

该项目的目标不是直接“全面替换原网络”，而是构建一套 **渐进式、可消融、易复现** 的实验框架，快速验证该机制在 DeepVIDv2 中是否成立。

---

## 2. 核心假设

### 2.1 背景动机

DeepVIDv2 的去噪任务同时依赖：

* 浅层特征中的局部高频与噪声统计
* 中层特征中的时空结构
* 深层特征中的重建上下文

标准 residual block 默认采用固定 identity/additive aggregation：

[
y = x + F(x)
]

这隐含假设：不同深度的 earlier representations 对当前层贡献是固定且等价的。

但对时空去噪而言，这一假设未必成立。不同层捕获的信息异质性很强，某些层可能更有利于保留快速动态，另一些层更有利于抑制噪声。

### 2.2 研究假设

引入 depth-wise attention aggregation 后，网络可对 earlier features 做选择性加权：

[
\tilde{x}*l = \sum*{i \in \mathcal{H}*l} \alpha*{i \to l} x_i, \quad \sum_i \alpha_{i \to l}=1
]

从而：

* 更好平衡低层细节与高层上下文
* 减少无效或有害的固定残差累积
* 提升空间去噪与时间保真的综合表现

### 2.3 预期趋势

我们预计性能不会随注入强度单调上升，而是呈现：

* 轻度或中度注入优于 baseline
* 过强注入收益递减，甚至损害稳定性与细节

因此主研究问题是：

> **AttnRes 在 DeepVIDv2 中的最优注入强度在哪里？**

---

## 3. 项目范围与非目标

### 3.1 项目范围

本 proposal 聚焦于：

* 在现有 DeepVIDv2 backbone 上做最小侵入式改造
* 实现多种 AttnRes 注入策略
* 保持训练流程基本不变
* 输出可复现实验与初步结论

### 3.2 非目标

本阶段不做：

* 全新 backbone 重写
* 数据集重构
* 与大量外部 SOTA 大规模 benchmark
* 复杂神经架构搜索
* 论文级 figure polish

目标是 **快速打样验证可行性**，而非一次性做完最终论文系统。

---

## 4. 设计原则

1. **最小侵入**：尽量不改 DeepVIDv2 主训练流程与数据管线。
2. **保守增量**：先局部替换，再逐步扩大注入范围。
3. **可消融**：每种改法都必须有明确实验编号与可对照实现。
4. **保稳定**：所有 AttnRes 模块保留原始 identity path，并引入可学习 gate。
5. **可配置**：所有注入位置、history 长度、温度、gate 初值都通过配置文件控制。
6. **快速回退**：任意实验失败时，能一键退回 baseline。

---

## 5. 模块设计

## 5.1 标准 residual block

原始形式：

[
y = x + F(x)
]

其中 `F(x)` 为当前 block 的主变换。

## 5.2 提议的 Gated Local AttnRes

建议首先实现如下保守版本：

[
y = x + F(x) + \gamma \cdot A(x_{l-K}, \dots, x_{l-1}; x_l)
]

其中：

* `A(...)`：对最近 K 个历史 block 输出做 depth-wise attention aggregation
* `\gamma`：可学习标量或通道级 gate，初始化为 0 或较小值
* `K`：history 长度，控制注入强度与开销

### 5.2.1 聚合模块输入输出

输入：

* 当前 stage 内最近 K 个 hidden states，shape 兼容当前 block
* 当前 block 输入 `x`

输出：

* 一个与 `x` 同 shape 的聚合特征图

### 5.2.2 score 生成方式（优先简单实现）

推荐首版使用：

* 对每个历史特征做 `GAP -> Linear` 得到 score
* 或 `1x1 conv -> GAP -> Linear`
* 对 depth 维度做 softmax
* 按权重求和得到聚合结果

即 attention 作用在 **层维度**，不是 token 维度。

### 5.2.3 训练稳定性保护

必须实现以下保护：

* 保留原始 identity `x`
* gate 初始化小值
* history 长度受限
* softmax temperature 可配置
* 默认只在局部 stage 内聚合，不做全网 full history

---

## 6. 代码架构 Proposal

## 6.1 新增模块

建议新增如下文件：

* `models/attnres.py`

  * `DepthAttentionAggregator`
  * `GatedAttnResidual`
  * `AttnResConfig`

* `models/blocks_attnres.py`

  * `AttnResBlockWrapper`
  * 若 DeepVIDv2 原本有统一 residual block，则直接 wrap

* `configs/ablation/*.yaml`

  * 各实验配置

## 6.2 关键类设计

### `AttnResConfig`

字段建议：

* `enabled: bool`
* `mode: str`  # off / bottleneck / bottleneck_decoder / stagewise / global
* `history_len: int`
* `temperature: float`
* `gate_init: float`
* `score_fn: str`  # gap_linear / conv1x1_gap_linear
* `share_proj: bool`
* `detach_history: bool`  # 默认 false
* `max_history_per_stage: int`

### `DepthAttentionAggregator`

职责：

* 输入一个 list of feature maps
* 生成 depth-wise scores
* softmax 得到权重
* 输出加权和
* 支持导出 attention weights 供可视化分析

### `GatedAttnResidual`

职责：

* 接收当前输入 `x` 与 history
* 调用 aggregator
* 应用 gate
* 输出 residual augmentation

### `AttnResBlockWrapper`

职责：

* wrap 原 block
* 控制是否启用 attnres
* 控制当前 block 所属 stage
* 从 stage history manager 中读取可用历史特征

## 6.3 History 管理

必须实现一个轻量 history 管理机制：

* 每个 stage 维护一个队列
* 只保存最近 K 个 block 输出
* 进入新 stage 时清空历史
* decoder 与 encoder 可各自独立管理

不建议首版做全局跨 stage history。

---

## 7. 实验矩阵

## 7.1 主实验

### E0 — Baseline

* 原始 DeepVIDv2
* 不启用 AttnRes

### E1 — Bottleneck-only

* 只在 bottleneck 末端 1–2 个 block 引入 AttnRes
* `history_len = 2`
* `gate_init = 0`

### E2 — Full Bottleneck

* 整个 bottleneck 内 block 启用 AttnRes
* `history_len = 3`
* `gate_init = 0`

### E3 — Bottleneck + Decoder

* bottleneck 与 decoder residual blocks 启用 AttnRes
* skip connections 不改
* `history_len = 3`

### E4 — Stage-wise Wide Injection

* encoder / bottleneck / decoder 各 stage 内启用 local AttnRes
* 只做 stage-local history
* `history_len = 4`

## 7.2 对照实验

### C1 — Concat Fusion Control

* 与 E3 同位置接入 history
* 不用 attention
* 采用 concat + 1x1 conv 融合

### C2 — Gate-only Control

* 与 E3 同位置
* 不引入 history attention
* 仅增加 gated residual scale

### C3 — Channel Attention Control

* 与 E3 同位置
* 用常规 channel attention 代替 depth attention

## 7.3 可选扩展实验

### X1 — Temperature Sweep

* `temperature = {0.5, 1.0, 2.0}`

### X2 — History Length Sweep

* `history_len = {2, 3, 4}`

### X3 — Gate Init Sweep

* `gate_init = {0, 1e-2, 1e-1}`

首轮打样建议优先跑 E0–E3 + C1。

---

## 8. 实验配置优先级

## Phase 1：最小可运行版本

必须完成：

* E0
* E1
* E2
* E3

目标：判断该方向是否成立。

## Phase 2：机制归因

必须完成：

* C1
* C2
* 可选 C3

目标：确认增益是否来自 depth-wise attention 本身。

## Phase 3：趋势与最优点

可选完成：

* E4
* X1 / X2 / X3

目标：找注入强度最优点与稳定边界。

---

## 9. 评估指标

## 9.1 核心性能指标

至少记录：

* PSNR
* SSIM
* temporal consistency / temporal fidelity
* 与原始 trace 的相关性指标

若数据管线支持，还应记录：

* voltage transient 保真度
* spike/event detectability 相关指标
* 去噪前后峰值形态保真

## 9.2 工程代价指标

必须记录：

* 参数量
* FLOPs 或近似计算量
* peak GPU memory
* 每 step 训练耗时
* 推理延迟

## 9.3 训练稳定性指标

必须记录：

* 收敛曲线
* 不同随机种子下方差
* attention 权重分布熵
* 是否出现 attention collapse

---

## 10. 输出物要求

Agent 最终需要输出：

### 10.1 代码输出

* 可运行的 AttnRes 模块实现
* 可插拔的 block wrapper
* 配置文件
* 训练脚本兼容 baseline 与 ablation

### 10.2 实验输出

* 每个实验的日志目录
* 指标汇总 CSV / JSON
* 参数量与速度统计
* attention weights dump（用于后续可视化）

### 10.3 文档输出

* README: 如何运行每个实验
* 简短 design note: 模块接入点与配置解释

---

## 11. Agent 实现任务拆分

## Task A — 代码审计与接入点定位

目标：

* 找出 DeepVIDv2 中 residual block 定义位置
* 标注 encoder / bottleneck / decoder 的 block 组织方式
* 确定最小插桩点

输出：

* `ARCHITECTURE_NOTES.md`
* block 拓扑说明
* 建议接入点列表

## Task B — AttnRes 最小模块实现

目标：

* 实现 `DepthAttentionAggregator`
* 实现 `GatedAttnResidual`
* 单元测试 shape 正确性与前向通过

输出：

* 模块代码
* smoke test

## Task C — Block Wrapper 接入

目标：

* 将 AttnRes 以 wrapper 形式接入原 residual block
* 保证 `enabled=False` 时完全退化为 baseline

输出：

* 兼容原训练脚本的模型构造接口

## Task D — 配置系统

目标：

* 新增 YAML 配置项
* 用实验名区分 E0/E1/E2/E3/C1...

输出：

* baseline 与 ablation configs

## Task E — 训练与评估脚本适配

目标：

* 自动记录参数量、显存、速度
* 自动导出主要指标

输出：

* 统一 runner
* metrics summary

## Task F — 快速实验首轮

目标：

* 先跑 E0/E1/E2/E3
* 输出初步对比表

输出：

* `results_phase1.csv`
* 简短结论 memo

---

## 12. Agent 实施约束

1. 不要大规模重构原工程。
2. 优先 wrap 而非重写主干。
3. 每一步都保证 baseline 可回退。
4. 所有新功能必须有配置开关。
5. 首轮不要实现 full global AttnRes。
6. 首轮不要修改 skip connection。
7. 首轮不要同时改 loss 或数据采样逻辑。
8. 所有实验命名统一、目录规范。

---

## 13. 目录结构建议

```text
project/
  models/
    attnres.py
    blocks_attnres.py
  configs/
    baseline.yaml
    ablation/
      e1_bottleneck.yaml
      e2_full_bottleneck.yaml
      e3_bottleneck_decoder.yaml
      c1_concat_control.yaml
      c2_gate_only.yaml
  scripts/
    train.py
    eval.py
    profile_model.py
  outputs/
    E0/
    E1/
    E2/
    E3/
    C1/
```

---

## 14. 风险与应对

## 风险 1：训练不稳定

表现：

* loss 波动明显
* attention 权重塌缩
* 结果劣于 baseline

应对：

* 降低注入强度
* 减小 history_len
* gate 初始化为 0
* 调高 softmax temperature

## 风险 2：显存/速度代价过大

表现：

* 训练变慢明显
* OOM

应对：

* 仅保留 stage-local history
* 限制 K
* 历史特征先做轻量投影
* 优先 bottleneck-only 版本

## 风险 3：效果提升不显著

表现：

* PSNR/SSIM 提升很小

应对：

* 检查是否只看空间指标，忽略时间指标
* 检查 gate 是否始终接近 0
* 引入更合适的注入位置对比
* 增加对照组，确认方向本身是否无效

## 风险 4：改动太多难归因

应对：

* 每次只改变一个因素
* 严格按照 E0 -> E1 -> E2 -> E3 顺序推进

---

## 15. 预期结果形式

理想情况下，最终将得到如下结论模式之一：

### 模式 A：轻中度注入最优

* E1 优于 E0
* E2 最优
* E3 持平或略优
* E4 收益递减

这将支持：

> 适度的 depth-wise adaptive residual aggregation 有利于 DeepVIDv2 去噪，但过度注入会带来冗余与稳定性问题。

### 模式 B：仅特定位置有效

* bottleneck 有效
* encoder 无效或有害

这将支持：

> AttnRes 主要对高语义/重建阶段有帮助，不适合浅层特征路径。

### 模式 C：总体无明显收益

这同样是有价值结论：

> 对 DeepVIDv2 而言，标准 residual 已足够，depth-wise attention aggregation 并未带来性价比优势。

---

## 16. 推荐首轮执行顺序

Agent 推荐按以下顺序实现与跑通：

1. 审计 DeepVIDv2 block 结构
2. 实现最小 `DepthAttentionAggregator`
3. 实现 `GatedAttnResidual`
4. 接入 E1（bottleneck-only）
5. 确保 baseline 与 E1 均能训练
6. 扩展到 E2、E3
7. 加入 C1 对照
8. 输出首轮表格与日志

---

## 17. 给 Agent 的一句话任务定义

> 在不破坏 DeepVIDv2 原有训练与数据流程的前提下，实现一个可配置、可回退、可消融的 Gated Local Attention-Residual 框架，并优先完成 E0/E1/E2/E3/C1 五组实验的快速打样。

---

## 18. 交付标准

本 proposal 完成的最低标准是：

* baseline 可正常复现
* E1/E2/E3 至少能前向与训练启动
* 配置切换无需改代码
* 实验输出可自动汇总
* 能回答“AttnRes 是否值得继续深挖”

本 proposal 完成的理想标准是：

* 获得一条清晰的注入强度-性能曲线
* 确认最优接入位置
* 为后续论文写作提供稳定实验骨架

