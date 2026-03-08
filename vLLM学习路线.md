# vLLM 专项学习路线

> 专注大模型推理引擎的工程实践

## 学习目标

掌握 vLLM 的部署、优化和二次开发能力

---

## 阶段一：入门（1周）

### 1.1 了解 vLLM 是什么

| 资源 | 链接 |
|-----|------|
| vLLM 官网 | https://docs.vllm.ai/ |
| GitHub | https://github.com/vllm-project/vllm |
| PagedAttention 论文 | https://arxiv.org/abs/2309.06180 |

**核心概念：**
- **PagedAttention**：类比操作系统分页内存管理，解决 LLM 显存问题
- **Continuous Batching**：动态批处理，提高 GPU 利用率
- **OpenAI API 兼容**：快速集成现有应用

---

### 1.2 本地部署第一个模型

**前置要求：**
```bash
# 安装（需要 CUDA 12.1+）
pip install vllm

# 或从源码安装
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

**快速启动：**
```bash
# 命令行方式（推荐先体验）
vllm serve Qwen/Qwen2-0.5B-Instruct

# OpenAI API 兼容模式
vllm serve Qwen/Qwen2-0.5B-Instruct \
    --api-key token-123 \
    --host 0.0.0.0 \
    --port 8000
```

**测试接口：**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-0.5B-Instruct",
    "prompt": "你好，请介绍一下自己",
    "max_tokens": 100
  }'
```

---

## 阶段二：核心功能（2周）

### 2.1 Python API 编程

```python
# basic_inference.py
from vllm import LLM, SamplingParams

# 初始化（首次运行自动下载模型）
llm = LLM(
    model="Qwen/Qwen2-0.5B-Instruct",
    tensor_parallel_size=1,  # 多卡并行
    dtype="half",            # FP16 推理
    gpu_memory_utilization=0.9,  # GPU 显存利用率
)

# 推理参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop=["\n"]  # 停止词
)

# 批量推理
prompts = [
    "写一个Python快排函数",
    "解释什么是REST API",
    "用Python写一个计算器"
]

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"=== Prompt {i+1} ===")
    print(output.outputs[0].text)
    print(f"Token usage: {output.usage}")
```

---

### 2.2 模型支持与选择

| 模型系列 | 示例模型 | 特点 |
|---------|---------|------|
| Qwen | Qwen/Qwen2-1.5B-Instruct | 中文友好，开源量大 |
| LLaMA | meta-llama/Llama-2-7b-chat-hf | 生态丰富 |
| Mistral | mistralai/Mistral-7B-Instruct-v0.2 | 性能优秀 |
| Mixtral | mistralai/Mixtral-8x7B-Instruct-v0.1 | MoE架构 |

**模型下载：**
```python
# 自动从 HuggingFace 下载
# 或使用镜像加速
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

llm = LLM(model="Qwen/Qwen2-1.5B-Instruct")
```

---

### 2.3 量化推理

```bash
# 命令行量化
vllm serve Qwen/Qwen2-1.5B-Instruct --quantization awq
vllm serve Qwen/Qwen2-1.5B-Instruct --quantization gptq
```

```python
# Python API 指定量化
llm = LLM(
    model="Qwen/Qwen2-1.5B-Instruct",
    quantization="awq",
    dtype="half",
)
```

**量化效果对比：**

| 量化方式 | 精度损失 | 显存减少 | 速度提升 |
|---------|---------|---------|---------|
| FP16 | - | 1x | 1x |
| INT8 (AWQ) | ~1% | ~2x | ~1.5x |
| INT4 (AWQ) | ~2-3% | ~4x | ~2x |

---

### 2.4 OpenAI API 兼容

```python
# openai_client.py
from openai import OpenAI

client = OpenAI(
    api_key="token-123",
    base_url="http://localhost:8000/v1"
)

# Chat API
chat_response = client.chat.completions.create(
    model="Qwen/Qwen2-0.5B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个专业助手"},
        {"role": "user", "content": "解释一下什么是Python"}
    ],
    temperature=0.7,
    max_tokens=256,
)

print(chat_response.choices[0].message.content)

# Completion API
completion = client.completions.create(
    model="Qwen/Qwen2-0.5B-Instruct",
    prompt="Python的优势是",
    max_tokens=100,
)

print(completion.choices[0].text)
```

---

## 阶段三：进阶与优化（2周）

### 3.1 多卡并行推理

```python
# multi_gpu_inference.py
from vllm import LLM

# Tensor Parallelism - 多卡分片
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    tensor_parallel_size=4,  # 使用 4 张 GPU
    dtype="half",
)

# 单卡放不下大模型时使用
# 7B 模型 → 2 卡
# 72B 模型 → 8 卡
```

**模型所需显存估算：**
```
显存 ≈ (模型参数量 × 2 bytes) × 1.2 (推理开销)
例: 7B 模型 ≈ 7 × 2 × 1.2 ≈ 16.8 GB (FP16)
```

---

### 3.2 服务化部署

```python
# api_server.py - 完整 API 服务
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import Optional, List, Dict
import uvicorn
import threading

app = FastAPI(title="vLLM API Service")

llm = None
lock = threading.Lock()

# ========== 数据模型 ==========
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "Qwen/Qwen2-0.5B-Instruct"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    model: str
    choices: List[dict]
    usage: dict

# ========== 初始化 ==========
@app.on_event("startup")
async def startup_event():
    global llm
    llm = LLM(
        model="Qwen/Qwen2-0.5B-Instruct",
        tensor_parallel_size=1,
        dtype="half",
    )

# ========== 路由 ==========
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    # 将消息转为 prompt
    prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages])
    prompt += "\nassistant:"

    sampling_params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    with lock:  # 防止并发冲突
        outputs = llm.generate([prompt], sampling_params)

    output = outputs[0]
    return ChatResponse(
        id=f"chatcmpl-{output.request_id}",
        model=req.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": output.outputs[0].text},
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": output.metrics.prompt_num_tokens,
            "completion_tokens": output.metrics.completion_num_tokens,
            "total_tokens": output.metrics.prompt_num_tokens + output.metrics.completion_num_tokens
        }
    )

@app.get("/models")
async def list_models():
    return {
        "data": [
            {"id": "Qwen/Qwen2-0.5B-Instruct", "object": "model"}
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 3.3 Docker 部署

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# 安装 Python 和依赖
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install vllm fastapi uvicorn

# 复制代码
COPY api_server.py /app/

EXPOSE 8000

CMD ["python3", "api_server.py"]
```

```bash
# 构建和运行
docker build -t vllm-api:latest .
docker run -d --gpus all -p 8000:8000 vllm-api:latest
```

---

### 3.4 性能调优参数

```python
# 性能优化配置
llm = LLM(
    model="Qwen/Qwen2-1.5B-Instruct",

    # 显存相关
    gpu_memory_utilization=0.95,    # GPU 显存利用率
    max_num_seqs=256,               # 最大并发序列数

    # 推理优化
    enforce_eager=False,             # False = CUDA Graph（更快）
    enable_chunked_prefill=False,    # 分块预填充

    # 量化
    quantization="awq",
    dtype="half",

    # 多卡
    tensor_parallel_size=1,
)
```

**关键参数说明：**

| 参数 | 说明 | 建议值 |
|-----|------|-------|
| `gpu_memory_utilization` | 显存使用比例 | 0.9-0.95 |
| `max_num_seqs` | 最大并发 | 根据显存调整 |
| `enforce_eager` | CUDA Graph | 生产环境 False |
| `enforce_fused_kv_cache` | 融合 KV Cache | True |

---

## 阶段四：实战项目（持续）

### 项目 1：本地问答系统

```python
# qa_system.py
from vllm import LLM, SamplingParams
from typing import List, Dict
import json

class QASystem:
    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B-Instruct"):
        self.llm = LLM(model=model_name, dtype="half")
        self.system_prompt = """你是一个专业的AI助手。
请根据下面的上下文回答用户问题。
如果无法从上下文找到答案，请如实说明。"""

    def answer(self, question: str, context: str = "") -> str:
        if context:
            prompt = f"上下文：{context}\n\n问题：{question}\n\n回答："
        else:
            prompt = f"问题：{question}\n\n回答："

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

# 使用
qa = QASystem()
answer = qa.answer("Python的装饰器是什么？")
print(answer)
```

---

### 项目 2：批量处理服务

```python
# batch_processor.py
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchProcessor:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=256,
        )

    def process_batch(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]

# 使用
processor = BatchProcessor("Qwen/Qwen2-0.5B-Instruct")
results = processor.process_batch([
    "什么是AI？",
    "Python有哪些优点？",
    "解释深度学习",
])
```

---

### 项目 3：LangChain 集成

```python
# langchain_integration.py
from langchain_community.llms import Vllm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# vLLM 作为 LangChain 的后端
llm = VLLM(
    model="Qwen/Qwen2-0.5B-Instruct",
    tensor_parallel_size=1,
    dtype="half",
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="用一句话介绍{topic}："
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("人工智能")
print(result)
```

---

## 学习检查清单

- [ ] 本地安装 vLLM 并成功运行
- [ ] 使用命令行启动模型服务
- [ ] 调用 OpenAI API 完成推理
- [ ] 使用 Python API 编写推理脚本
- [ ] 实现量化模型加载（AWQ/INT8）
- [ ] 完成 FastAPI 服务化部署
- [ ] 使用 Docker 容器化部署
- [ ] 完成至少 1 个实战项目

---

## 常见问题

**Q: 模型下载很慢怎么办？**
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

**Q: 显存不够怎么办？**
1. 换更小的模型（如 0.5B/1.5B）
2. 使用量化（INT8/INT4）
3. 减少 `gpu_memory_utilization`

**Q: 推理速度慢怎么办？**
1. 开启 CUDA Graph (`enforce_eager=False`)
2. 增大 `max_num_seqs`
3. 使用量化模型
4. 多卡并行

**Q: 如何查看 GPU 利用率？**
```bash
watch -n 1 nvidia-smi
```
