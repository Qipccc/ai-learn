# 大模型工程实践学习路线

> 适用于有编程基础（Python），想转型大模型工程方向的开发者

## 目录

- [阶段二：大模型工程核心](#阶段二大模型工程核心)
  - [1. 模型部署](#1-模型部署)
  - [2. 模型优化](#2-模型优化)
  - [3. GPU 基础](#3-gpu-基础)
- [阶段三：分布式与系统工程](#阶段三分布式与系统工程)
  - [1. 分布式训练](#1-分布式训练)
  - [2. 集群管理](#2-集群管理)
- [阶段四：实战项目](#阶段四实战项目)
- [推荐资源](#推荐资源)

---

## 阶段二：大模型工程核心

### 1. 模型部署

#### vLLM（强烈推荐必学）

| 资源 | 链接 |
|-----|------|
| 官方文档 | https://docs.vllm.ai/ |
| GitHub | https://github.com/vllm-project/vllm |
| 快速入门教程 | https://docs.vllm.ai/en/latest/getting_started/quickstart.html |

**实战代码 - 本地部署开源模型：**

```python
# vllm_server.py
from vllm import LLM, SamplingParams

# 加载模型（支持 Qwen, LLaMA, Mistral 等）
llm = LLM(
    model="Qwen/Qwen2-0.5B-Instruct",
    tensor_parallel_size=1,
    dtype="half",
)

# 推理
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
)

prompts = ["写一个 Python 快排函数", "解释一下什么是 REST API"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

**启动命令：**

```bash
# CLI 方式
vllm serve Qwen/Qwen2-0.5B-Instruct --dtype half

# OpenAI API 兼容方式
vllm serve Qwen/Qwen2-0.5B-Instruct --dtype half --api-key token-abc123
```

---

#### Triton Inference Server（NVIDIA 出品）

| 资源 | 链接 |
|-----|------|
| 官方文档 | https://docs.nvidia.com/deeplearning/triton-inference-server/ |
| GitHub | https://github.com/triton-inference-server/server |
| Python 客户端 | https://github.com/triton-inference-server/client |

**实战 - 部署模型仓库：**

```bash
# 1. 拉取 Triton 镜像
docker pull nvcr.io/nvidia/tritonserver:24.02-py3

# 2. 准备模型仓库目录结构
# model_repository/
# └── qwen2/
#     └── 1/
#         └── model.py (TensorRT-LLM 或 PyTorch 模型)

# 3. 启动服务
docker run --gpus=1 -p8000:8000 -p8001:8001 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.02-py3 \
  tritonserver --model-repository=/models
```

---

#### FastAPI 构建模型服务

| 资源 | 链接 |
|-----|------|
| 官方文档 | https://fastapi.tiangolo.com/ |
| 视频教程 | https://www.youtube.com/watch?v=0sOvCIG3gKU |

**实战代码 - 模型 API 服务：**

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn

app = FastAPI(title="LLM API")

llm = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    usage: dict

@app.on_event("startup")
async def startup():
    global llm
    llm = LLM(model="Qwen/Qwen2-0.5B-Instruct")

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    sampling_params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    outputs = llm.generate([req.prompt], sampling_params)
    return GenerateResponse(
        text=outputs[0].outputs[0].text,
        usage={"prompt_tokens": 0, "completion_tokens": 0}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 2. 模型优化

#### 量化（Quantization）

| 技术 | 文档 | 代码 |
|-----|------|------|
| AWQ | https://github.com/mit-han-lab/awq | - |
| GPTQ | https://github.com/autodl/GPTQ | - |
| GGUF (llama.cpp) | https://github.com/ggerganov/llama.cpp | - |
| BMMX (vLLM内置) | https://docs.vllm.ai/en/latest/quantization.html | vLLM 已支持 |

**实战 - 使用 GPTQ 量化模型：**

```python
# 方式1: 使用 transformers + auto-gptq
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import ExllamaGPTQQuantizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
quantizer = ExllamaGPTQQuantizer(
    bits=4,
    group_size=128,
    desc_act=False,
)
quantizer.quantize_model(model)
model.save_pretrained("qwen2-0.5b-4bit")
```

**实战 - vLLM 加载量化模型：**

```bash
# vLLM 支持多种量化格式
vllm serve Qwen/Qwen2-0.5B-Instruct --quantization awq
```

---

#### 推理优化 - PagedAttention

| 资源 | 链接 |
|-----|------|
| 论文 | https://arxiv.org/abs/2309.06180 |
| vLLM 原理讲解 | https://docs.vllm.ai/en/latest/architecture.html |

---

#### LoRA 微调（工程视角）

| 资源 | 链接 |
|-----|------|
| LLaMA-Factory | https://github.com/Kuangddx/LLaMA-Factory |
| PEFT (HuggingFace) | https://github.com/huggingface/peft |
| 文档 | https://huggingface.co/docs/peft/index |

**实战 - 使用 LLaMA-Factory 微调：**

```bash
# 单卡微调
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset alpaca_zh \
    --finetuning_type lora \
    --output_dir qwen_lora \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3
```

---

### 3. GPU 基础

#### CUDA 基础概念（工程视角）

| 资源 | 链接 |
|-----|------|
| NVIDIA CUDA 入门 | https://developer.nvidia.com/cuda-education |
| CUDA C++ 教程 | https://www.parallel-forall.com/ |

**GPU 监控命令：**

```bash
# 查看 GPU 状态
nvidia-smi

# 实时监控
watch -n 1 nvidia-smi

# 查看 GPU 进程
fuser -v /dev/nvidia*
```

---

## 阶段三：分布式与系统工程

### 1. 分布式训练

#### DeepSpeed（微软分布式训练库）

| 资源 | 链接 |
|-----|------|
| 官方文档 | https://www.deepspeed.ai/getting-started/ |
| GitHub | https://github.com/microsoft/DeepSpeed |
| 快速入门 | https://www.deepspeed.ai/getting-started/quickstart/ |

**实战 - DeepSpeed 训练配置：**

```json
// ds_config.json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "fp16": {
        "enabled": "auto"
    },
    "zero_allow_untested_optimizer": true
}
```

**启动训练：**

```bash
deepspeed train.py \
    --deepspeed ds_config.json \
    --num_gpus 8
```

**实战 - ZeRO 优化原理（必知）：**

- ZeRO-1: 优化器分片
- ZeRO-2: 优化器 + 梯度分片
- ZeRO-3: 优化器 + 梯度 + 参数分片

---

#### FSDP (Fully Sharded Data Parallel)

| 资源 | 链接 |
|-----|------|
| PyTorch FSDP 教程 | https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html |
| 官方文档 | https://pytorch.org/docs/stable/fsdp.html |

---

### 2. 集群管理

#### Kubernetes (K8s)

| 资源 | 链接 |
|-----|------|
| K8s 官方文档 | https://kubernetes.io/zh-cn/docs/home/ |
| Kubeflow | https://www.kubeflow.org/ (ML 任务调度) |

**实战 - GPU 调度配置：**

```yaml
# gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

---

#### Slurm 调度器

| 资源 | 链接 |
|-----|------|
| Slurm 文档 | https://slurm.schedmd.com/documentation.html |

**实战 - 提交 GPU 任务：**

```bash
#!/bin/bash
# job.sh
#SBATCH --job-name=llm_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

srun python train.py --deepspeed ds_config.json
```

```bash
sbatch job.sh
```

---

## 阶段四：实战项目

### 项目 1：搭建本地模型服务

```bash
# 1. 安装 vllm
pip install vllm

# 2. 启动服务
vllm serve Qwen/Qwen2-1.5B-Instruct --dtype half

# 3. 测试 API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-1.5B-Instruct",
    "prompt": "你好，请介绍一下自己",
    "max_tokens": 100
  }'
```

### 项目 2：模型量化部署

```bash
# 使用 AWQ 量化
pip install awq

# 量化模型
python -m awq.entry --model Qwen/Qwen2-1.5B-Instruct --output ./qwen-awq

# 启动量化后模型
vllm serve ./qwen-awq --quantization awq
```

### 项目 3：构建 API 服务

```bash
# 1. 启动 vLLM API 服务
vllm serve Qwen/Qwen2-1.5B-Instruct --api-key token-123

# 2. 部署 FastAPI 代理（可添加鉴权、限流等功能）

# 3. Docker 容器化
docker build -t llm-api:latest .
docker run -d -p 8000:8000 --gpus all llm-api:latest
```

---

## 推荐资源

### 文档

- [vLLM 官方文档](https://docs.vllm.ai/)
- [NVIDIA Triton 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [DeepSpeed 官方教程](https://www.deepspeed.ai/getting-started/)
- [PyTorch 分布式文档](https://pytorch.org/docs/stable/distributed.html)
- [Kubernetes 官方文档](https://kubernetes.io/zh-cn/docs/home/)

### 开源项目

| 项目 | 说明 |
|-----|------|
| vLLM | 高性能 LLM 推理引擎 |
| DeepSpeed | 分布式训练框架 |
| Hugging Face Transformers | 模型库 |
| LLaMA-Factory | 简易微调框架 |
| llama.cpp | 高效推理引擎 |

### 进阶方向

| 方向 | 技能点 | 岗位示例 |
|-----|-------|---------|
| 模型部署 | vLLM、Triton、API 开发 | ML Platform Engineer |
| 推理优化 | 量化、蒸馏、CUDA | Performance Engineer |
| 训练系统 | DeepSpeed、K8s、调度 | ML Infrastructure Engineer |
| 数据管道 | Spark、RAPIDS、ETL | Data Engineer |

---

## 学习路线图

```
阶段二（2-3个月）
├── vLLM 部署 → 1周
├── FastAPI 服务 → 1周
├── 模型量化 → 1周
└── 推理优化 → 2周

阶段三（2-3个月）
├── DeepSpeed 训练 → 2周
├── FSDP 分布式 → 1周
├── K8s 基础 → 2周
└── 集群调度 → 1周

阶段四（持续）
├── 项目实战
└── 参与开源
```
