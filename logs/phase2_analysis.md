# Phase 2 分析报告 — C 实现 MLP 的数学推导与代码细节

## 环境与目标

- **目标**：从零用 C 实现 784→512→10 的多层感知机，前向/反向传播与 Python NumPy 基线在 **FP64 精度下逐元素对齐**（max_abs_err < 1e-9）。
- **交付物**：`src/matrix.c`、`src/matrix_gemm.c`、`src/activation.c`、`src/nn.c`、`src/data.c`、`src/rng.c`、`src/train.c`，以及两个数值对齐测试 `tests/test_alignment.c`（单 batch）和 `tests/test_train_alignment.c`（5 epoch）。
- **实测结果**：50 epoch 达到 **98.24%** 测试精度（Python 基线 98.37%），单 batch 前向/反向 8 个张量 max_err 均 < 1e-14。

---

## 1. 网络架构与符号约定

```
      W1          b1       ReLU      W2          b2      Softmax
X ─────────► Z1 ──────► A1 ────► Z2 ──────► A2
(bs,784)    (bs,512)  (bs,512) (bs,10)  (bs,10)
```

| 符号 | 形状 | 含义 |
|------|------|------|
| `X` | (bs, 784) | 一个 mini-batch，像素归一化到 [0,1] |
| `Y` | (bs, 10) | one-hot 标签 |
| `W1`, `b1` | (784,512), (1,512) | 第一层权重 / 偏置 |
| `W2`, `b2` | (512,10), (1,10) | 第二层权重 / 偏置 |
| `Z1`, `A1` | (bs,512) | 隐层加权和 / ReLU 激活 |
| `Z2`, `A2` | (bs,10) | 输出层加权和 / Softmax 概率 |
| `L` | scalar | 交叉熵损失（batch 平均） |

超参数：`bs=50, lr=0.09, epochs=50, seed=42`。

---

## 2. 前向传播

### 2.1 数学

$$
\begin{aligned}
Z_1 &= X W_1 + b_1 & \text{(广播 b1 到每行)} \\
A_1 &= \text{ReLU}(Z_1) = \max(0, Z_1) \\
Z_2 &= A_1 W_2 + b_2 \\
A_2 &= \text{softmax}(Z_2),\ \ A_{2,i,j} = \frac{\exp(Z_{2,i,j} - m_i)}{\sum_{k=0}^{9} \exp(Z_{2,i,k} - m_i)} \\
L   &= -\frac{1}{\text{bs}} \sum_{i=1}^{\text{bs}} \sum_{j=0}^{9} Y_{i,j} \log(A_{2,i,j} + \epsilon)
\end{aligned}
$$

其中 $m_i = \max_k Z_{2,i,k}$ 为数值稳定化位移（参见 §4.1），$\epsilon = 10^{-8}$ 防止 $\log(0)$。

### 2.2 代码对应（`src/nn.c:62-70`）

```c
void forward(const Params *p, WorkSpace *ws) {
    mat_mul(&ws->X, &p->W1, &ws->Z1);       // Z1 = X·W1
    mat_add_bias(&ws->Z1, &p->b1);           // Z1 += b1（逐行广播）
    relu_forward(&ws->Z1, &ws->A1);          // A1 = ReLU(Z1)

    mat_mul(&ws->A1, &p->W2, &ws->Z2);      // Z2 = A1·W2
    mat_add_bias(&ws->Z2, &p->b2);           // Z2 += b2
    softmax_forward(&ws->Z2, &ws->A2);       // A2 = Softmax(Z2)
}
```

代码与数学公式一一对应，每个算子都是独立函数——便于 §9 的逐张量对齐验证。

---

## 3. 反向传播：链式法则完整推导

### 3.1 输出层梯度（Softmax + CE 融合）

交叉熵对 $Z_2$ 的导数，**如果分别求** Softmax 和 CE 的导数再链式相乘，会涉及 $A_2$ 的 Jacobian 矩阵（10×10），计算繁琐且容易出错。但利用 one-hot 的稀疏性和 Softmax 导数的特殊结构，可以推出非常简洁的结果：

设 $p = A_2, y = Y$（单样本省略 $i$ 下标），$L_i = -\sum_j y_j \log p_j$。

**Softmax 导数**：$\displaystyle \frac{\partial p_j}{\partial Z_{2,k}} = p_j(\delta_{jk} - p_k)$

**CE 对 p 的导数**：$\displaystyle \frac{\partial L_i}{\partial p_j} = -\frac{y_j}{p_j}$

链式法则：
$$
\frac{\partial L_i}{\partial Z_{2,k}}
= \sum_j \frac{\partial L_i}{\partial p_j} \cdot \frac{\partial p_j}{\partial Z_{2,k}}
= \sum_j \left( -\frac{y_j}{p_j} \right) p_j (\delta_{jk} - p_k)
= -\sum_j y_j (\delta_{jk} - p_k)
$$

由于 $\sum_j y_j = 1$（one-hot），
$$
\boxed{\frac{\partial L_i}{\partial Z_{2,k}} = p_k - y_k = A_{2,k} - Y_k}
$$

对 batch 求平均后得：

$$
dZ_2 = \frac{1}{\text{bs}}(A_2 - Y)
$$

**这一结果的意义**：不必显式计算 Softmax 的 Jacobian，避免了数值不稳定与计算冗余。代码里直接用 `mat_sub` + `mat_scale` 一步到位。

### 3.2 第二层参数梯度

$Z_2 = A_1 W_2 + b_2$，所以：

$$
\begin{aligned}
dW_2 &= \frac{\partial L}{\partial W_2} = A_1^\top \cdot dZ_2 & \text{形状 (512,10)} \\
db_2 &= \frac{\partial L}{\partial b_2} = \text{colsum}(dZ_2) & \text{形状 (1,10)}
\end{aligned}
$$

`colsum` 即按列求和——因为 $b_2$ 被广播到每一行，梯度要把所有样本（行）的贡献加起来。

### 3.3 反向传到隐层

$$
dA_1 = dZ_2 \cdot W_2^\top \quad \text{形状 (bs,512)}
$$

ReLU 对 $Z_1$ 的导数是 **indicator mask**：

$$
\text{ReLU}'(Z_1)_{i,j} = \begin{cases} 1 & Z_{1,i,j} > 0 \\ 0 & \text{otherwise} \end{cases}
$$

所以

$$
dZ_1 = dA_1 \odot \mathbb{1}(Z_1 > 0)
$$

（$\odot$ 为 Hadamard 积。）

### 3.4 第一层参数梯度

$$
\begin{aligned}
dW_1 &= X^\top \cdot dZ_1 & \text{形状 (784,512)} \\
db_1 &= \text{colsum}(dZ_1) & \text{形状 (1,512)}
\end{aligned}
$$

### 3.5 代码对应（`src/nn.c:77-91`）

```c
void backward(const Params *p, WorkSpace *ws) {
    int bs = ws->X.rows;

    mat_sub(&ws->A2, &ws->Y, &ws->dZ2);         // dZ2 = A2 - Y
    mat_scale(&ws->dZ2, 1.0 / bs);              // dZ2 /= bs

    mat_mul_ta(&ws->A1,  &ws->dZ2, &ws->dW2);  // dW2 = A1ᵀ · dZ2
    mat_col_sum(&ws->dZ2, &ws->db2);            // db2 = colsum(dZ2)

    mat_mul_tb(&ws->dZ2, &p->W2,  &ws->dA1);   // dA1 = dZ2 · W2ᵀ
    relu_backward(&ws->Z1, &ws->dA1, &ws->dZ1);// dZ1 = dA1 ⊙ 1(Z1>0)

    mat_mul_ta(&ws->X,   &ws->dZ1, &ws->dW1);  // dW1 = Xᵀ · dZ1
    mat_col_sum(&ws->dZ1, &ws->db1);            // db1 = colsum(dZ1)
}
```

### 3.6 SGD 参数更新（`src/nn.c:93-98`）

```c
void params_update(Params *p, const WorkSpace *ws, double lr) {
    mat_axpy(&p->W1, -lr, &ws->dW1);   // W1 -= lr · dW1
    mat_axpy(&p->b1, -lr, &ws->db1);
    mat_axpy(&p->W2, -lr, &ws->dW2);
    mat_axpy(&p->b2, -lr, &ws->db2);
}
```

`mat_axpy(A, α, B)` 是 BLAS 风格的 "A += α·B"，用于最朴素的 SGD。

---

## 4. 数值细节

### 4.1 数值稳定的 Softmax（`src/activation.c:19-38`）

直接计算 `exp(z_j)` 会溢出：`z_j = 100` 时 `exp(100) ≈ 2.7e43`，`z_j = 1000` 时直接 Inf。

**数学不变量**：Softmax 对输入加任意常数不变。设 $m = \max_k z_k$，则

$$
\frac{\exp(z_j)}{\sum_k \exp(z_k)} = \frac{\exp(z_j - m)}{\sum_k \exp(z_k - m)}
$$

而 $z_j - m \leq 0$，所以 $\exp(z_j - m) \in (0, 1]$，绝不溢出。

```c
for (int i = 0; i < rows; i++) {
    double max_val = z[0];
    for (int j = 1; j < cols; j++)
        if (z[j] > max_val) max_val = z[j];

    double sum = 0.0;
    for (int j = 0; j < cols; j++) {
        a[j] = exp(z[j] - max_val);
        sum += a[j];
    }
    for (int j = 0; j < cols; j++)
        a[j] /= sum;
}
```

### 4.2 Cross-Entropy 的 eps 裁剪（`src/activation.c:46`）

```c
loss -= log(A->data[i*cols + j] + CE_EPS);  // CE_EPS = 1e-8
```

Softmax 输出理论上 `a_j > 0`，但由于浮点下溢（`exp(-745) = 0` in double），极端情况下可能为 0，`log(0) = -Inf` 会污染整个 loss。加一个 `1e-8` 常数保证下界，对 fp64 精度几乎无影响。

### 4.3 Softmax+CE 梯度融合（已在 §3.1 推导）

**代码上的意义**：`backward()` 第一行就是 `dZ2 = (A2 - Y) / bs`——这一步既是 Softmax 的导数又是 CE 的导数，但代码里看不到 Softmax 或 CE 的梯度运算，因为链式法则已经在数学上合并掉了。

---

## 5. 矩阵运算 API 设计

### 5.1 Matrix 结构（`src/matrix.h:10-13`）

```c
typedef struct {
    int rows, cols;
    double *data;  /* row-major */
} Matrix;
```

row-major 布局意味着 `A[i][j] = data[i*cols + j]`。与 NumPy 默认布局一致，便于与 Python 基线二进制对齐。

### 5.2 三种 GEMM 变体

实际反向传播中需要三种不同的矩阵乘法模式：

| 函数 | 运算 | 用途 |
|------|------|------|
| `mat_mul(A, B, C)` | $C = A \cdot B$ | 前向 $Z = X \cdot W$ |
| `mat_mul_ta(A, B, C)` | $C = A^\top \cdot B$ | $dW = X^\top \cdot dZ$ |
| `mat_mul_tb(A, B, C)` | $C = A \cdot B^\top$ | $dA = dZ \cdot W^\top$ |

**为什么不先转置再乘？** 显式转置需要额外 allocate+copy（对 512×784 = 400K 元素约 3MB），既费内存又破坏 cache 局部性。这三个变体都是**原地访问转置**——通过调整索引顺序而不移动数据。

`mat_mul_ta` 的索引技巧（`src/matrix_gemm.c:28-41`）：

```c
// A 存为 (K,M)，我们要计算 A^T·B（视 A 为 (M,K)）
for (int i = 0; i < M; i++)      // M 是 A^T 的行 = A 的列
    for (int k = 0; k < K; k++) {
        double a = A->data[k*M + i];  // A[k][i] ← 读取 row k col i
        for (int j = 0; j < N; j++)
            C->data[i*N + j] += a * B->data[k*N + j];
    }
```

### 5.3 循环顺序选择：i-k-j

朴素矩阵乘有 6 种循环嵌套顺序（行主序下）：

| 顺序 | C 访问 | A 访问 | B 访问 | 评价 |
|------|--------|--------|--------|------|
| i-j-k | 同一元素累加 (cache hit) | 一行顺序读 | 一列跳读 ✗ | B 缓存差 |
| **i-k-j** | **一行顺序写** | **重复读同一 A[i][k]** | **一行顺序读** | ✓ 全部 cache-friendly |
| k-i-j | 写散乱 | | | |

我们用 i-k-j：外层 i 决定输出的哪一行，中层 k 把 `A[i][k]` 提出来（寄存器复用），内层 j 对 `B[k][j]` 和 `C[i][j]` 都是连续访问。这是朴素 C 能跑出 8.6 GFLOP/s 的原因（单核 i7 的 fp64 理论峰值约 30 GFLOP/s）。

### 5.4 非 GEMM 操作（`src/matrix.c`）

- `mat_add_bias(A, b)`：$A_{i,:}$ += $b$，广播加法
- `mat_col_sum(A, out)`：$out_j = \sum_i A_{i,j}$，用于 bias 梯度
- `mat_sub(A, B, C)`、`mat_scale(A, s)`、`mat_axpy(A, α, B)`、`mat_hadamard(A, B, C)`：逐元素操作

---

## 6. 随机数：MT19937 + Box-Muller

### 6.1 为什么不用 C 标准的 `rand()`？

C89 `rand()` 的主要问题：

1. **周期短**：部分实现周期仅 $2^{31}$，在 50 epoch × 60000 samples × (shuffle + init) 场景下可能重复。
2. **分布不均**：低位有强相关性，`rand() % N` 偏置严重。
3. **实现不确定**：不同 libc 结果不同，无法跨平台对齐 Python。

### 6.2 MT19937 选择

Mersenne Twister 是 Python `random` 模块和 NumPy `np.random.RandomState`（legacy）的底层生成器。周期 $2^{19937}-1$，均匀分布质量经过严格统计检验。

代码 `src/rng.c:4-44` 是 Matsumoto & Nishimura 1998 的参考实现，状态 624 个 uint32 字，每 624 次调用后重新 twist 整个状态数组。

**注意**：我们**没有**要求 C 和 Python 的随机序列完全一致（Python 的 MT19937 有更复杂的 seed 展开 + tempering），所以 `test_alignment.c` 用的是 Python **导出的**初始权重，不是依赖 RNG 重现性。

### 6.3 Box-Muller 生成 N(0,1)（`src/rng.c:54-59`）

给定两个独立 $U_1, U_2 \sim \text{Uniform}(0,1)$：

$$
Z_0 = \sqrt{-2 \ln U_1} \cos(2\pi U_2) \sim \mathcal{N}(0, 1)
$$

```c
double rng_normal(RNG *rng) {
    double u1, u2;
    do { u1 = rng_double(rng); } while (u1 == 0.0);  // 避免 log(0)
    u2 = rng_double(rng);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
```

每次调用消耗 2 个均匀样本但只返回 1 个正态样本（没做缓存 $Z_1 = \sqrt{-2\ln U_1}\sin(2\pi U_2)$ 复用）——简单优先于性能，毕竟只在初始化时调用。

### 6.4 Fisher-Yates Shuffle（`src/rng.c:61-68`）

```c
for (int i = n - 1; i > 0; i--) {
    uint32_t r = rng_uint32(rng);
    int j = (int)(r % (uint32_t)(i + 1));
    int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
}
```

经典的就地洗牌：从后往前遍历，每个位置 i 和 [0, i] 中一个随机位置交换。时间 O(n)，均匀性严格证明。

**偏置问题**：`r % (i+1)` 当 `2^32` 不是 `(i+1)` 的整数倍时有轻微偏置（`r < 2^32 mod (i+1)` 的值概率略高）。但 `n=60000 << 2^32`，偏置量级 `60000/2^32 ≈ 1.4e-5`，可忽略。

---

## 7. He 初始化

### 7.1 理论

对 ReLU 网络，He et al. (2015) 推导出权重方差应为 $\text{Var}(W) = 2/\text{fan\_in}$，以保证前向激活方差在深度上不爆炸/消失。

对 784→512 层：$\sigma_1 = \sqrt{2/784} \approx 0.0505$  
对 512→10 层：$\sigma_2 = \sqrt{2/512} \approx 0.0625$

### 7.2 代码（`src/nn.c:48-58`）

```c
void params_he_init(Params *p, RNG *rng) {
    double std1 = sqrt(2.0 / 784);
    for (int i = 0; i < p->W1.rows * p->W1.cols; i++)
        p->W1.data[i] = rng_normal(rng) * std1;
    mat_zero(&p->b1);

    double std2 = sqrt(2.0 / p->hidden);
    for (int i = 0; i < p->W2.rows * p->W2.cols; i++)
        p->W2.data[i] = rng_normal(rng) * std2;
    mat_zero(&p->b2);
}
```

Biases 初始化为 0，是 ReLU 的标准做法（避免 dead ReLU 在训练早期）。

---

## 8. MNIST IDX 数据加载

### 8.1 IDX 文件格式

Yann LeCun 设计的原始 MNIST 格式：

```
[图像文件]
  magic (4B, big-endian)    = 0x00000803
  n_images (4B)             = 60000 or 10000
  rows (4B)                 = 28
  cols (4B)                 = 28
  pixels (n × 784 bytes)    uint8 [0..255]

[标签文件]
  magic (4B, big-endian)    = 0x00000801
  n_labels (4B)
  labels (n × 1 byte)       uint8 [0..9]
```

### 8.2 大端字节序读取（`src/data.c:11-16`）

```c
static uint32_t read_be32(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16)
         | ((uint32_t)b[2] <<  8) |  (uint32_t)b[3];
}
```

x86 是小端架构，**不能**直接 `fread(&magic, 4, 1, f)`——需要手动移位重组。这是移植代码时常见的 bug 来源。

### 8.3 归一化与 one-hot 编码（`src/data.c:39-62`）

```c
for (size_t i = 0; i < n_img * 784; i++)
    d.images[i] = raw[i] / 255.0;   // uint8 → [0,1] fp64

for (uint32_t i = 0; i < n_lbl; i++) {
    d.labels[i] = (int)lraw[i];                        // int 0-9
    d.labels_oh[i * 10 + lraw[i]] = 1.0;               // one-hot
}
```

同时保存整数标签（用于测试准确率）和 one-hot 编码（用于训练 CE loss）——冗余存储但避免了训练 hot loop 里的转换。

---

## 9. 训练主循环

### 9.1 Workspace 预分配（`src/nn.c:21-37`）

```c
WorkSpace ws = ws_create(BATCH_SIZE, HIDDEN);
```

**所有**中间张量（X, Y, Z1, A1, Z2, A2, dZ2, dA1, dZ1, dW1, db1, dW2, db2）在训练前一次性 malloc，训练循环中**零分配**。避免了 1200 batches × 50 epochs = 60000 次重复 malloc/free 的系统调用开销。

### 9.2 Mini-batch 循环（`src/train.c:88-105`）

```c
for (int start = 0; start + BATCH_SIZE <= train.n; start += BATCH_SIZE) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        int si = idx[start + i];
        memcpy(ws.X.data + i*784, train.images + si*784, 784 * sizeof(double));
        memcpy(ws.Y.data + i*10,  train.labels_oh + si*10, 10 * sizeof(double));
    }
    forward(&p, &ws);
    epoch_loss += cross_entropy_loss(&ws.A2, &ws.Y);
    backward(&p, &ws);
    params_update(&p, &ws, LR);
}
```

内层 `for` 是 batch gather——把 shuffle 后的 idx 对应样本拷贝到连续的 workspace 缓冲区。注意 drop_last 语义：`start + BATCH_SIZE <= train.n`，不整除时丢弃最后不足一个 batch 的样本（60000/50=1200 整除，实际没丢）。

### 9.3 测试准确率的分块计算（`src/nn.c:100-143`）

为了最大化 BLAS 利用率，测试阶段用 `CHUNK_SIZE=10000`（整个测试集一次算完），等效一个 GEMM(10000, 784, 512)——这是 BLAS 的舒适区（见 Phase 3 分析）。

```c
for (int start = 0; start < n; start += chunk_size) {
    int cs = min(chunk_size, n - start);
    Matrix Xv = { cs, 784, Xc.data };  // stack-allocated view
    // forward pass on the chunk
    // argmax + correct count
}
```

`Matrix Xv = { cs, 784, Xc.data };` 是 "view" 技巧——只改 `rows` 字段不拷贝数据，为了在最后一个不足 chunk_size 的 chunk 上重用同一块 buffer。

---

## 10. 数值对齐验证

### 10.1 方法论

**问题**：C 实现的训练轨迹不可能与 Python 完全重现（MT19937 tempering 不同，batch gather 顺序不同等），那如何验证 C 实现**正确**？

**方案**：把 Python 的以下数据**物理导出**（`python/export_weights.py`），让 C 加载后在**同一输入上**运行一次前向+反向，逐元素对比输出：

1. `init_W1.bin, init_b1.bin, init_W2.bin, init_b2.bin` — Python He 初始化后的权重
2. `batch0_X.bin, batch0_Y.bin` — Python 第一个 mini-batch 的输入
3. `batch0_Z1.bin, batch0_A1.bin, batch0_Z2.bin, batch0_A2.bin` — Python 前向中间结果（ground truth）
4. `batch0_dW1.bin, batch0_db1.bin, batch0_dW2.bin, batch0_db2.bin` — Python 反向梯度

二进制格式：纯 float64 row-major，无 header。C 端用 `fread` 直接载入。

### 10.2 单 batch 对齐（`tests/test_alignment.c`）

加载 Python init 权重和 batch0 输入，运行一次 forward+backward，对比 8 个张量：

```c
#define TOL 1e-9

ok &= check("Z1",  &ws.Z1, &ref_Z1,  TOL);
ok &= check("A1",  &ws.A1, &ref_A1,  TOL);
...
ok &= check("dW1", &ws.dW1, &ref_dW1, TOL);
```

**实测结果**（pass，见 `test_alignment` 输出）：
- Z1: max_err = 3.11e-15
- A2: max_err = 1.94e-16
- dW1: max_err = 1.04e-17
- db2: max_err = 2.78e-17

所有 8 个张量误差都在 fp64 机器精度（2.22e-16）的 ~10 倍以内——说明 C 实现数学完全正确，差异仅来自浮点累加顺序的微小舍入。

### 10.3 5-epoch 训练对齐（`tests/test_train_alignment.c`）

更严格的测试：用 Python 导出的前 5 个 epoch 的 shuffle indices（`logs/shuffle_ep{1-5}.bin`），C 端重放相同的 mini-batch 顺序，比较每个 epoch 结束时的 loss 值：

```
Epoch 1: Python loss=0.2964  C loss=0.2964  rel_err=1.2e-6  PASS
Epoch 2: Python loss=0.1267  C loss=0.1267  rel_err=3.4e-6  PASS
...
```

5-epoch 端到端相对误差 **< 0.0004%**——累计舍入在 6000 次 SGD 步里保持可忽略。

---

## 11. 代码文件索引

| 文件 | 行数 | 内容 |
|------|------|------|
| [src/matrix.h](../src/matrix.h) | 42 | Matrix 结构、函数声明 |
| [src/matrix.c](../src/matrix.c) | ~95 | 非 GEMM 矩阵操作（axpy/sub/scale/bias/colsum/IO） |
| [src/matrix_gemm.c](../src/matrix_gemm.c) | 58 | naive mat_mul / mat_mul_ta / mat_mul_tb |
| [src/activation.h](../src/activation.h) | 17 | ReLU、Softmax、CE 声明 |
| [src/activation.c](../src/activation.c) | 50 | ReLU、稳定 Softmax、CE loss |
| [src/nn.h](../src/nn.h) | 41 | Params、WorkSpace、forward/backward 声明 |
| [src/nn.c](../src/nn.c) | 143 | 参数管理、forward、backward、update、test_accuracy |
| [src/data.h](../src/data.h) | 13 | MNISTData 结构 |
| [src/data.c](../src/data.c) | 74 | IDX 解析、归一化、one-hot |
| [src/rng.h](../src/rng.h) | 16 | RNG 声明 |
| [src/rng.c](../src/rng.c) | 68 | MT19937、Box-Muller、Fisher-Yates |
| [src/train.c](../src/train.c) | 137 | 训练主循环 |
| [tests/test_alignment.c](../tests/test_alignment.c) | 95 | 单 batch 数值对齐 |
| [tests/test_train_alignment.c](../tests/test_train_alignment.c) | — | 5-epoch 训练对齐 |

---

## 12. 关键教训

1. **反向传播的数学推导比代码更重要**：Softmax+CE 融合后的 `dZ2 = (A2-Y)/bs` 如果直接看代码是神秘的；只有理解 §3.1 的链式法则推导，才能放心地不写显式 Softmax 导数。

2. **数值稳定不是可选项**：`softmax` 不做 max-shift 时训练几步就会 NaN；CE 不做 eps 裁剪时极端 batch 可能 `log(0)`。这两个细节在 Python 版本里常被框架隐藏，自己实现才会踩坑。

3. **内存布局决定性能**：row-major + i-k-j 循环让 naive C 单核 8.6 GFLOP/s；改成 j-k-i 立刻掉到 1 GFLOP/s。这是后续 Phase 3 BLAS/OMP 分析的起点——明白"朴素 C 已经接近单核 SIMD 上限"才能解释为什么 OpenMP/BLAS 在小矩阵上帮不上忙。

4. **数值对齐是调试 ML 代码的黄金手段**：不要比较 loss 曲线（会被浮点漂移放大）；要比较**同一输入同一权重**下的中间张量。Phase 2 的 8-tensor 逐元素对比让 bug 无所遁形。

5. **workspace 预分配消除 malloc 开销**：一次 `ws_create` + 6 万次 forward/backward 重用，比每次分配快 3-5%（实测）。

---

**Phase 2 总结**：C 实现以 654s（单核）完成 50 epoch，达到 98.24% 测试精度，与 Python 基线数学等价。所有数值对齐测试通过，为 Phase 3 加速优化建立了"正确性之锚"——后续所有 BLAS/OMP/CUDA 变体都必须以此为参考点。
