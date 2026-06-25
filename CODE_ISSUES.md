# TATN 代码问题记录

对照论文（Shen et al., IEEE TPEL 2023）整理的代码问题，分三类：真实 Bug、自动化不足、论文与实现不符。

---

## 一、真实 Bug

### 1. `pretrain.py` — 预训练模型从不被保存

**位置**：`pretrain.py` 第 84-85 行

```python
#save_model(models, optimizers, loss_min, seed,path)
#print('min loss:{} saved model'.format(loss_min))
```

`save_model` 被注释掉，预训练跑完后最优模型不会写入磁盘。`./models/` 目录下的 `.pt` 文件是作者自己保存好的，不是跑这份代码生成的。

---

### 2. `pretrain.py` — best-model 追踪条件永远为 False

**位置**：`pretrain.py` 第 114 行

```python
if epoch > epochs:   # epoch 最大为 epochs-1，此条件永远不成立
```

这段本应追踪测试集上最优模型的逻辑从未被执行过。

---

### 3. `train.py` — 每个 batch 都 print，输出爆炸

**位置**：`train.py` 第 109-110 行

```python
print(target_domain_pred.detach().cpu().numpy()[0])
print(source_domain_pred.detach().cpu().numpy()[0])
```

训练循环的每个 mini-batch 都会打印两行，epoch 数 × batch 数量的输出会让 Colab/终端极度卡顿。

---

### 4. `models.py` — `load_saved_model` 硬编码 CUDA，忽略传入的 device 参数

**位置**：`models.py` 第 147 行

```python
def load_saved_model(device, models, optimizers, loss_min, seed, model_path='./saved_model/best.pt'):
    device = torch.device('cuda')   # 覆盖了传入的 device 参数
```

在 CPU 环境（或非 cuda:0 的设备）下直接报错。正确写法应保留传入的 `device` 参数。

---

## 二、自动化不足（非逻辑错误，但需手动操作）

### 5. `normalized_data/process.py` — 归一化范围硬编码，每个温度需手动修改

**位置**：`process.py` 第 111-114 行

```python
def transform(path):
    range_current = [-17.6, 6.003]
    range_voltage = [2.799, 4.209]
    range_temp    = [23.555, 26.82]   # 仅适用于 25°C 数据！
    range_ah      = [-2.591, 0.00131]
    # above range need to be revised according to different file
```

**正确流程**（作者未说明，需手动执行）：

1. 先对某个温度目录跑 `read_mat(path)` → 生成 `data.txt`，记录该温度的全局 min/max
2. 手动将四个 range 修改为 `data.txt` 中对应的值
3. 再跑 `transform(path)` 归一化该温度数据
4. **每换一个温度重复一遍**

若跳过此步骤，直接用 25°C 的范围处理其他温度数据，会导致特征值严重超出 `[0, 1]`（例如 10°C 的温度值归一化后约为 -4.15），模型输入分布完全错误。

---

## 三、论文描述与代码实现不符

### 6. Transfer 阶段：source model 并未完全冻结，且 LSTM 被共享

**论文**（Section III.B, Fig. 1）：

> *"at the transfer stage, only the feature extractor and the domain discriminator of the target model are trainable, while the remaining parts in source and target models are frozen."*

即 source model（Conv_s + LSTM_s + FC_s + Regression_s）应全部冻结，只训练 target 的 feature extractor 和 discriminator。

**代码实际情况**（`train.py`）：

- `models['lstm_s']` 在 `run.py` 中定义，但在 `train.py` 中**从未被使用**
- source 和 target **共用同一个 `models['lstm']`**，并且该 LSTM 在 transfer 阶段会被更新（`optimizers['lstm'].step()`）

```python
# train.py
source_features = models['lstm'](models['conv_s'](source_data))  # conv_s 独立，lstm 共享
target_features = models['lstm'](models['conv'](target_data))    # conv 独立，lstm 共享
...
optimizers['conv'].step()   # 只更新 target conv
optimizers['lstm'].step()   # 更新共享 lstm（source 和 target 都用的）
```

这与论文 Fig.1 的架构图有出入，实际上是 CNN 分离、LSTM 共享的结构，而不是论文所说的完全独立的 source/target feature extractor。

---

### 7. Discriminator 结构与论文不符

**论文**：discriminator 输入为 BiLSTM 输出的 feature（时序特征）。

**代码**（`models.py` `Discriminator.forward`）：

```python
def forward(self, x):
    # LSTM 层被注释掉，直接 flatten + FC
    # x = x.permute(0, 2, 1)
    # x, (h,c) = self.lstm1(x)
    # x, (h,c) = self.lstm2(x)
    x = torch.reshape(x, (x.shape[0], -1))
    x = self.fc2(x)
    return x
```

Discriminator 中的 LSTM 层全部注释掉，直接将 feature flatten 后接 FC 层做二分类，是更简化的实现。

---

## 总结

| 编号 | 文件 | 类型 | 严重程度 |
|------|------|------|----------|
| 1 | `pretrain.py` | Bug：模型不保存 | 高 |
| 2 | `pretrain.py` | Bug：条件永远 False | 中 |
| 3 | `train.py` | Bug：过量 print | 中 |
| 4 | `models.py` | Bug：硬编码 CUDA | 中 |
| 5 | `process.py` | 自动化不足：归一化范围需手动改 | 高（复现关键） |
| 6 | `train.py` | 与论文不符：LSTM 共享而非分离 | 低（不影响运行） |
| 7 | `models.py` | 与论文不符：Discriminator 简化 | 低（不影响运行） |
