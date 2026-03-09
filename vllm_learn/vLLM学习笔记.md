# vLLM 学习笔记

> 基于与 Claude 的对话整理

---

## 1. 基于 Transformer 的 LLM 是什么

**Transformer** 是 2017 年 Google 论文《Attention Is All You Need》提出的深度学习架构，核心是 **Self-Attention（自注意力）机制**。

### Transformer 基本结构

```
输入文本 → Tokenize → Embedding → Transformer Encoder/Decoder → Output
```

### LLM 的两种主要架构

| 架构 | 代表模型 | 特点 |
|-----|---------|------|
| **Decoder-only**（最常用） | GPT、LLaMA、Qwen | 单向注意力，只看前面的词 |
| **Encoder-Decoder** | T5、BART | 编码-解码结构 |

### 核心：Self-Attention

```python
# Attention 本质：计算每个词对其他词的重要性
# 公式：Attention(Q, K, V) = softmax(QK^T / √d) * V
```

---

## 2. Transformer 自回归分解公式

### 核心思想

LLM 的本质是 **建模下一个词的概率分布**：

$$
P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})
$$

### 分解公式

给定前 i-1 个词，预测第 i 个词的概率：

$$
P(w_i | w_{<i}) = \text{Softmax}(W \cdot \text{Transformer}(w_{<i}))
$$

### 生成时的自回归过程

```python
def generate(prompt, max_tokens=100):
    input_ids = tokenizer(prompt)

    for _ in range(max_tokens):
        # 前向传播
        logits = model(input_ids)

        # 只取最后一个词的 logits
        next_token_logits = logits[-1]

        # 采样（多种方式）
        next_token = sample(softmax(next_token_logits / temperature))

        # 终止条件
        if next_token == eos_token:
            break

        # 添加到输入序列
        input_ids.append(next_token)

    return tokenizer.decode(input_ids)
```

---

## 3. Token 是什么

**Token** 是 LLM 处理文本的最小单位，可以理解为"词"或"字符"。

### Tokenize（分词）

```python
# 示例：文字 → Token
text = "今天天气很好"
tokens = ["今天", "天气", "很好"]
token_ids = [1003, 2501, 5823]
```

### Token 的粒度

| 粒度 | 示例 | 优点 | 缺点 |
|-----|------|-----|------|
| **词** | "今天" | 语义明确 | 词表太大 |
| **字符** | "今", "天" | 词表小 | 语义弱 |
| **子词** (最常用) | "今天", "天" | 平衡 | 需要算法 |

### Token 相关计算

```python
# 1个 token ≈ 1-2 个英文单词 ≈ 0.5-2 个中文汉字
"Hello, how are you?"  → 5 tokens
"你好世界"              → 3 tokens
```

---

## 4. Token 与 KV Cache 的关系

### 核心概念

```
Token → 通过 Attention 计算 → 需要 Key (K) 和 Value (V)
                                ↓
                          缓存起来 = KV Cache
```

### Transformer Attention 计算

```python
# 标准 Attention 公式
# Attention(Q, K, V) = softmax(QK^T / √d) * V
```

### KV Cache 的作用

**问题**：生成第 N 个 token 时，需要计算与前 N-1 个 token 的 Attention

**KV Cache**：只计算新 token 的 K 和 V，之前的结果缓存起来

```python
# 有 KV Cache：增量计算
K_cache = []
V_cache = []

for new_token in generated_tokens:
    # 只计算新 token 的 K, V（恒定速度）
    K_new, V_new = compute_single_token(new_token)
    K_cache.append(K_new)
    V_cache.append(K_new)

    output = attention(Q_new, K_cache, V_cache)
```

### Token 数量与显存关系

```
KV Cache 显存 ≈ 2 × num_layers × batch_size × seq_len × hidden_size × dtype

例子：7B 模型，32K 上下文
- 13G 显存（FP16）
- 每个 token 约需 400KB KV Cache
```

---

## 5. vLLM 生成过程的两阶段

### 整体流程图

```
输入: "推荐一部电影"
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Prompt Phase (预填充阶段)                          │
│  ─────────────────────────────────────────────────────────  │
│  一次性处理整个 prompt，计算所有 token 的 K 和 V               │
│  "推荐" → [K₁,V₁]                                           │
│  "一部" → [K₂,V₂]                                           │
│  "电影" → [K₃,V₃]                                           │
│                              ↓                               │
│  输出第一个新 token: "《"                                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Autoregressive Generation (自回归生成)            │
│  ─────────────────────────────────────────────────────────  │
│  逐个生成新 token，每次只需要计算 1 个 token                  │
│  Step 1: "《" + KV Cache → 预测 "流浪"                     │
│  Step 2: "《流浪" + KV Cache → 预测 "地球"                  │
│  Step 3: "《流浪地球" + KV Cache → 预测 "》"                │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

### 两阶段对比

| 阶段 | 处理方式 | 计算特点 | 耗时占比 |
|-----|---------|---------|---------|
| **Prompt Phase** | 并行处理所有 prompt token | 计算密集，可批量 | 约 20-30% |
| **Autoregressive** | 逐个生成新 token | 每次 1 个 token | 约 70-80% |

---

## 6. K 和 V 是如何生成的

### K 和 V 的生成流程

```
输入 Token → Embedding → 线性变换 → K 和 V
```

### 详细代码

```python
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        # Q、K、V 的线性变换矩阵
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        # 线性变换生成 Q、K、V
        q = self.q_proj(hidden_states)  # Query
        k = self.k_proj(hidden_states)  # Key
        v = self.v_proj(hidden_states)  # Value

        # 计算 Attention
        output = self.attention(q, k, v)

        return output
```

### 多层堆叠：每层都生成新的 K 和 V

```
输入 Token → Embedding → Transformer Layer 1 → K₁, V₁
                               ↓
                           Transformer Layer 2 → K₂, V₂
                               ↓
                           ...
                               ↓
                           Transformer Layer N → K_N, V_N
```

### Attention 计算过程

```python
def attention(q, k, v):
    # 1. 计算 QK^T（相似度）
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / (head_dim ** 0.5)

    # 2. Softmax 归一化
    attn_weights = torch.softmax(scores, dim=-1)

    # 3. 加权求和
    output = torch.matmul(attn_weights, v)

    return output
```

---

## 7. PagedAttention 原理

### 核心思想

用操作系统的分页思想管理 KV Cache

| 传统方式 | PagedAttention |
|---------|----------------|
| 连续内存分配 | 非连续分页管理 |
| 显存碎片化 | 灵活分页，减少碎片 |
| 预分配固定大小 | 动态按需分配 |

### 图示

```
传统 KV Cache（连续）：
[Token 1] [Token 2] [Token 3] ... [Token N] [空闲]

PagedAttention（分页）：
Page 1: [Token 1-16]
Page 2: [Token 17-32]
Page 3: [Token 33-48]
...
```

---

## 8. Beam Search（束搜索）

### 什么是 Beam Search

**Beam Search** 是一种**贪心搜索策略**，在每一步保留概率最高的 Top-K 个候选序列，最终选择总分最高的序列。

### 对比：Greedy vs Beam Search

| 方法 | 原理 | 优点 | 缺点 |
|-----|------|-----|------|
| **Greedy** | 每次选概率最高的 1 个 | 速度快 | 可能错过最优解 |
| **Beam Search** | 保留 Top-K 个最优路径 | 更可能找到全局最优 | 速度较慢 |

### Beam Search 原理图解

```
假设 beam_size=2

Step 1:
"今天" → 概率: 0.4
"今天天气" → 概率: 0.3  ← 保留
"今天心情" → 0.2  ← 保留
"今天吃" → 0.1

Step 2:
"今天天气" → "今天天气好" 0.25  ← 保留
               "今天天气不" 0.05
"今天心情" → "今天心情好" 0.15  ← 保留
               "今天心情不" 0.05

Step 3: 计算总分，选择最高的
"今天天气好" = 0.3 × 0.25 = 0.075  ← 最终输出
"今天心情好" = 0.2 × 0.15 = 0.030
```

### Beam Search 代码实现

```python
def beam_search(model, prompt, beam_size=5, max_len=100):
    # 1. 初始化
    sequences = [(prompt, 0.0)]  # (序列, log概率)

    for step in range(max_len):
        all_candidates = []

        # 2. 对每个候选序列扩展
        for seq, score in sequences:
            # 预测下一个 token
            logits = model(seq)
            probs = softmax(logits[-1])

            # 3. 取 Top-K
            top_k_probs, top_k_ids = torch.topk(probs, beam_size)

            # 4. 生成新候选
            for i in range(beam_size):
                new_seq = seq + [top_k_ids[i]]
                new_score = score + log(top_k_probs[i])
                all_candidates.append((new_seq, new_score))

        # 5. 选择 Top-K 作为下一轮
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        sequences = all_candidates[:beam_size]

    # 6. 返回概率最高的
    return sequences[0][0]
```

### vLLM 中的 Beam Search

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2-0.5B-Instruct")

# Beam Search 配置
sampling_params = SamplingParams(
    best_of=5,           # beam_size
    max_tokens=100,
    use_beam_search=True,  # 启用束搜索
)

# vLLM 会自动进行 Beam Search
outputs = llm.generate(prompt, sampling_params)
```

### Beam Search vs 其他采样方法

| 方法 | 特点 | 适用场景 |
|-----|------|---------|
| **Greedy** | 确定性 | 需要确定输出的任务 |
| **Beam Search** | 兼顾多样性与质量 | 机器翻译、摘要 |
| **Top-K** | 随机采样 | 创意写作 |
| **Top-P (Nucleus)** | 动态阈值 | 对话生成 |
| **Temperature** | 控制随机性 | 调节多样性 |

### Beam Search 的问题

1. **计算量大**：需要维护多个候选序列
2. **显存占用**：beam_size 越大，显存开销越大
3. **可能重复**：容易陷入重复生成循环

---

## 相关资源

- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- 本地源码：`aiLearn/vllm_learn/code/vllm/`
- 本地论文：`aiLearn/vllm_learn/paper/paged_attention.pdf`
