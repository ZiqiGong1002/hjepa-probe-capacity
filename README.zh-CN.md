# H-JEPA FourRoom 探针：复现与探针容量研究

[English](README.md) · [上游原始 README](UPSTREAM_README.md)

[H-JEPA](https://arxiv.org/abs/2610.06805) 判断分层表征每一层还保留哪些信息，方法是在冻结的特征上训练一个小的探针网络。论文 Fig. 4 的结论是：越高的层越保留 agent 的位置，同时逐层丢掉干扰物的信息（"selective abstraction"）。论文只用了一种探针（隐藏层 512 的 MLP）。本研究检验这个结论是否依赖于探针的强度。

- **上游**：[kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA)，提交 `97054ec`（MIT）。官方 FourRoom checkpoint 取自 Hugging Face，已校验 sha256。
- **本人的改动**：上游之上的 3 个代码提交（`git log 97054ec..HEAD -- h_jepa`），以及 `probing/` 目录和本 README（官方原 README 保留为 [`UPSTREAM_README.md`](UPSTREAM_README.md)）。
- **不包含**：数据集和 checkpoint（按 [`UPSTREAM_README.md`](UPSTREAM_README.md) 下载）。

## 结论速览

1. **Fig. 4 复现成功。** 4 层模型、3 个模型种子下，各层干扰物位置误差为 0.053 / 0.167 / 0.340 / 0.553；论文图中读数约为 0.06 / 0.19 / 0.37 / 0.58。
2. **逐层趋势对各种宽度的 MLP 探针都成立。** 在 3 层模型上，隐藏层 128、512、2048 的 MLP 探针都显示干扰物误差从第 1 层到第 3 层递增；宽度超过 128 后，数值变化不超过 0.02。
3. **换成线性探针，结论弱很多。** 线性探针下干扰物误差为 0.45 / 0.67 / 0.90，因此「丢掉了多少信息」很大程度上取决于用哪类探针。
4. **这些信息来自训练。** 同架构、随机初始化的编码器，各层干扰物误差都在 0.78 左右，agent 位置误差约 0.91。
5. **仍有一处对不上。** 2 层模型（Fig. 12）的第 2 层与论文一致，第 1 层不一致（0.156 vs 约 0.01），原因未查明。

## 实验设置

- **模型**：官方 H-JEPA FourRoom checkpoint（2、3、4 层），冻结。
- **探针**：官方探针配置：AdamW，lr 1e-3，weight decay 1e-4，batch 128，60 epoch，探针种子 42。单隐藏层 + LayerNorm + GELU，只改变隐藏层宽度（0 = 线性）。
- **预测目标**：agent 位置（`xy`）、干扰物位置（`distractor0_xy`）、到最近干扰物的测地距离（`min_distractor_dist`）。
- **指标**：评测集上的归一化 MSE（NMSE）。1.0 表示不比预测均值好，越低越好。
- **硬件**：RTX 4080 Laptop GPU 12GB，WSL2。Python 3.11，torch 2.7.1+cu126。

### 代码改动（`h_jepa/` 内）

| 提交 | 改动 |
|---|---|
| Make probe hidden_dim configurable | 新增 `probe_hidden_dim` 选项；`0` 为线性；默认 512 时走原代码路径 |
| Add cached-embedding probing runner | `main_probing_cached.py`：冻结编码器只运行一次并缓存输出，再用相同的探针头、相同的随机数序列和指标训练探针 |
| Cached probing: optional random_init baseline | `+random_init=true`：用官方构造函数建同架构的未训练模型 |

**缓存版与官方是否等价**：在 3 层模型（seed 42，MLP512 探针）上，缓存版的 18 个 NMSE 与官方流程逐位相同（最大差值 0）。见 `probing/scripts/compare_metrics.py`。

## 结果

### 1. 论文 Fig. 4：4 层模型，干扰物位置

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| 论文（图中读数） | ≈0.06 ± 0.02 | ≈0.19 ± 0.05 | ≈0.37 ± 0.08 | ≈0.58 ± 0.07 |
| 本仓库，3 个模型种子，均值 ± 标准误 | 0.053 ± 0.020 | 0.167 ± 0.050 | 0.340 ± 0.073 | 0.553 ± 0.072 |

各层 agent 位置都几乎完美（NMSE ≈ 1e-4），与论文一致。

### 2. 论文 Fig. 12：2 层模型（seed 42）

| | agent L1 | agent L2 | 干扰物 L1 | 干扰物 L2 |
|---|---|---|---|---|
| 论文 | ≈0 | ≈0 | ≈0.01 | ≈0.33 ± 0.01 |
| 本仓库 | 4e-5 | 7e-5 | **0.156** | 0.341 |

第 2 层一致；第 1 层干扰物误差不一致，原因未找到。Fig. 12 图注称各环境的探针头不同，那里用的探针可能不是公开配置里的这一个。

### 3. 探针容量：3 层模型（seed 42）

论文没有报告 3 层模型的探针结果。

| 探针（参数量） | 干扰物位置 L1 / L2 / L3 | 测地距离 L1 / L2 / L3 | agent 位置 |
|---|---|---|---|
| 线性（386） | 0.446 / 0.672 / 0.903 | 0.816 / 0.854 / 0.863 | 0.002–0.007 |
| MLP 128（25k） | 0.117 / 0.313 / 0.507 | 0.326 / 0.506 / 0.631 | ≈1e-4 |
| MLP 512，官方（101k） | 0.105 / 0.275 / 0.492 | 0.307 / 0.489 / 0.619 | ≈1e-4 |
| MLP 2048（403k） | 0.098 / 0.257 / 0.487 | 0.301 / 0.467 / 0.615 | ≈1e-4 |

- 从线性到 128 是最大的一步（第 3 层干扰物误差下降 0.40）。128 以上基本饱和：第 3 层数值变化不超过 0.02，小于模型种子间的波动（Fig. 4 中约 0.07）。
- 所有探针都保持 L1 < L2 < L3 的顺序，因此 Fig. 4 的结论对 MLP 宽度是稳健的。但在线性探针下，连第 1 层看起来都丢掉了大量干扰物信息。
- 最宽的探针没有过拟合：训练集与评测集的差距（约 0.03）和 512 探针相同。
- 这组扫描只用了一个模型种子。

**对照：只用 agent 位置能预测多少？** 用同样的 MLP512 探针，输入换成真实 agent 位置加上关着的门的编号（不用任何学到的特征），干扰物误差为 0.981，测地距离误差为 0.873。学到的特征明显更好（测地距离 0.31–0.62），说明它们确实包含干扰物信息，而不只是通过 agent 位置间接推出来的。

### 4. 随机编码器基准（3 层架构，未训练，MLP512 探针）

| 评测 NMSE L1 / L2 / L3 | agent 位置 | 干扰物位置 | 测地距离 |
|---|---|---|---|
| 随机编码器 | 0.904 / 0.919 / 0.906 | 0.773 / 0.777 / 0.797 | 0.993 / 0.995 / 0.996 |
| 训练后的 H-JEPA | 5e-5 / 7e-5 / 7e-5 | 0.105 / 0.275 / 0.492 | 0.307 / 0.489 / 0.619 |

随机特征包含的信息很少，而且各层几乎一样，所以层的堆叠本身并不会丢信息。训练让 agent 位置在每一层都几乎完美，并让干扰物在第 1 层可读。越高的层，干扰物误差越向随机编码器的水平回落，但仍低于它，说明丢弃并不彻底。

## 预注册

在看到容量扫描的任何完整结果之前，我把预期写在了 [`probing/PREREG.md`](probing/PREREG.md) 里。结果对照：

| | 预期 | 结果 |
|---|---|---|
| H1 | agent 位置在所有容量下都可解码；线性明显更差；MLP 之间饱和 | **成立** |
| H2 | 干扰物位置在所有容量下都约为 1.0（「表征中不存在」） | **被推翻**：MLP 探针下为 0.10–0.49 |
| H3 | 测地距离主要由 agent 位置解释（表征 ≈ 对照） | **被推翻**：表征 0.30–0.62，对照 0.873 |
| H4 | 若最宽探针只改善训练集、不改善评测集，则判为「不可解码」 | 未触发：训练–评测差距不变 |

H2 和 H3 判断错了：特征里的干扰物信息比我预想的多。

## 复现步骤

```bash
# 1. 环境与数据：按 UPSTREAM_README.md（uv，Python 3.11）；然后
export HJEPA_HOME=/path/to/data

# 2. HF 上的 checkpoint 不带探针脚本需要的训练配置，需要重建，例如
python probing/scripts/make_train_config.py h_jepa/config/train fourroom_hjepa_l3 42 \
    $HJEPA_HOME/ckpts/fourroom/fourroom_hjepa_l3/seed42/config.yaml

# 3. 单次探针运行：<模型> <模型种子> <探针隐藏层宽度>
bash probing/scripts/run_probe.sh l3 42 512

# 本文所有运行，依次执行
bash probing/scripts/sweep.sh

# 4. 只用 agent 位置的对照
python probing/scripts/xy_only_control.py
```

`probing/results/` 里是本文报告的每次运行的 `metrics.yaml`。3 层模型的线性探针一行来自官方流程（`results/C_h0`），其余容量来自缓存版。

## 说明

- 在测试机器上（WSL2 下的笔记本 GPU），同时运行两个 CUDA 进程必定崩溃（Windows 在崩溃时刻记录了 `nvlddmkm` "Graphics FECS Exception" 事件）。本文所有运行都是串行的，`run_probe.sh` 会对失败的运行重试最多 3 次。
- torch 用的是 cu126，而不是上游 README 写的 cu128，因为测试机器访问不了官方 wheel 源。

## 下一步

- FourRoom 里没有腿姿这类快速变化的身体变量。要检验「高层丢掉腿姿等快变量、保留位置」这一说法，需要在具身环境中验证（H-JEPA 也提供了 Ant 环境）。

## 致谢

模型与训练代码均出自 H-JEPA 作者。如使用，请引用：

```bibtex
@article{zhang2026hjepa,
  title   = {{H-JEPA}: End-to-End Learning of Hierarchical World Models for Visual Planning},
  author  = {Zhang, Wancong and Terver, Basile and Rabbat, Mike and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint arXiv:2610.06805},
  year    = {2026}
}
```

## 许可证

MIT，沿用上游，见 [LICENSE](LICENSE)。
