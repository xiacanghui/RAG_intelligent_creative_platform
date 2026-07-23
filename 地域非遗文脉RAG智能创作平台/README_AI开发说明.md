# 地域非遗文脉RAG智能创作平台 - 项目开发说明

## 一、项目概述

地域非遗文脉RAG智能创作平台是一个基于检索增强生成（RAG）技术的智能创作辅助系统，专注于非物质文化遗产的保护、传承与创新应用。本平台采用轻量化B/S架构，支持离线运行，适用于文旅非遗产品开发、文化宣传内容创作等场景。

## 二、核心技术

### 1. RAG检索增强生成技术
- 基于ChromaDB向量数据库实现本地知识库存储
- 采用LangChain框架构建标准RAG检索链路
- 支持非遗文本的智能切片、向量化和相似度检索

### 2. 双模型调度架构
- 离线主模型：阿里开源qwen3.6，通过Ollama本地部署，支持RTX4060 GPU加速推理
- 线上备用模型：讯飞星火V3云端API，优化长文案生成效果
- 自动网络检测与模型切换，确保服务可用性

### 3. 国风智能前端
- 基于HTML5+TailwindCSS v3的轻量化前端
- 国风米白、赭石、青蓝低饱和配色
- 支持全屏演示，无冗余弹窗

## 三、功能模块

### 1. 非遗智能问答
- 支持800字以内的长文本输入
- 基于RAG技术的精准非遗知识检索
- 展示匹配的非遗史料来源和工艺分类标签

### 2. 传统纹样生成
- 内置皮影、竹编、年画、剪纸、木雕、湘绣、苏绣等非遗品类
- 基于用户描述生成标准化国风纹样AI绘图提示词

### 3. 批量文创内容生成
- 内置4套固定行业模板：
  - 景区线下导览解说文案
  - 非遗科普短视频完整脚本
  - 文创商品宣传介绍文案
  - 游客国风朋友圈打卡文案

### 4. 知识库管理
- 支持批量上传本地TXT非遗文档
- 自动切片、向量化入库
- 可视化展示知识库条目，支持单条删除

### 5. 内容导出工具
- 单段文本一键复制
- 批量生成内容一键打包下载为TXT文件

## 四、技术架构

```
┌─────────────────────────────────────────────────┐
│                    前端层                        │
│  HTML5 + TailwindCSS v3 + 原生JavaScript        │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│                  API网关层                       │
│              FastAPI + CORS                     │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ RAG核心  │  │ 模型调度  │  │ 内容过滤  │
│ LangChain│  │ 双模型   │  │ 合规检查  │
└──────────┘  └──────────┘  └──────────┘
        │             │
        ▼             ▼
┌──────────┐  ┌──────────┐
│ ChromaDB │  │ Ollama   │
│ 向量存储  │  │ 本地推理  │
└──────────┘  └──────────┘
```

## 五、硬件要求

- 内存：16GB DDR4
- 显卡：RTX4060 Laptop 8GB独显（GPU加速推理）
- 存储：至少10GB可用空间

## 六、软件依赖

- Python 3.10+
- Ollama（本地大模型管理）
- Node.js（可选，用于前端开发）

## 七、快速开始

### 1. 环境准备
```bash
# 安装Python依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.template .env
# 编辑.env文件，填写讯飞星火API密钥（可选）
```

### 2. 启动Ollama服务
```bash
# 安装Ollama（如果未安装）
# 访问 https://ollama.ai 下载安装

# 拉取qwen3模型
ollama pull qwen3:latest

# 启动Ollama服务
ollama serve
```

### 3. 启动应用服务
```bash
# 启动FastAPI服务
python main.py
```

### 4. 访问应用
打开浏览器，访问 http://localhost:8000

## 八、配置说明

### 环境变量配置（.env文件）
```bash
# 讯飞星火API配置（可选）
SPARK_APPID=your_app_id
SPARK_API_KEY=your_api_key
SPARK_API_SECRET=your_api_secret

# Ollama配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:latest

# 服务配置
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# 存储路径
CHROMA_PERSIST_DIR=./vector_store
KNOWLEDGE_BASE_DIR=./knowledge_base
LOG_DIR=./logs
```

## 九、目录结构

```
地域非遗文脉RAG智能创作平台/
├─ main.py                  # FastAPI主服务入口
├─ ollama_run.py            # Ollama配置和启动脚本
├─ model_switch.py          # 双模型调度控制器
├─ rag_core.py              # RAG检索核心逻辑
├─ utils/
│  ├─ token_control.py      # Token节流管控工具
│  ├─ style_filter.py       # 非遗风格校验过滤工具
│  └─ content_check.py      # 通用内容合规过滤
├─ static/                  # 前端页面文件
│  └─ index.html            # 单页交互前端页面
├─ knowledge_base/          # 本地非遗素材库
├─ vector_store/            # ChromaDB向量存储
├─ archive/                 # 大赛申报归档文件夹
├─ requirements.txt         # 依赖清单
├─ .env.template            # 环境变量模板
└─ README.md                # 项目说明文档
```

## 十、开源组件声明

本项目使用以下开源组件：

| 组件名称 | 开源协议 | 用途 |
|---------|---------|------|
| FastAPI | MIT License | Web服务框架 |
| LangChain | MIT License | RAG链式调用框架 |
| ChromaDB | Apache License 2.0 | 向量数据库 |
| Ollama | MIT License | 本地大模型管理 |
| TailwindCSS | MIT License | CSS样式框架 |
| Qwen3 | Apache License 2.0 | 开源大语言模型 |

## 十一、合规声明

1. 本项目全程不接入任何境外大模型服务
2. 所有非遗素材、对话缓存均本地持久化存储
3. 无数据出境风险，符合数据安全法规要求
4. 代码头部统一声明：基础工程代码由Vibe Coding智能体生成；非遗分类加权检索算法、Token分层节流缓存系统、非遗风格强制校验过滤引擎三大核心业务算法模块由项目负责人独立人工重构开发

## 十二、联系方式

项目负责人：[姓名]
联系电话：[电话]
电子邮箱：[邮箱]

---

> 本说明文档可直接复制粘贴至大赛申报表相关栏目
