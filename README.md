# 智能文档问答系统 - 功能与使用说明书

## 一、系统概述

智能文档问答系统是一个基于 RAG（Retrieval-Augmented Generation，检索增强生成）技术的文档问答平台。系统支持上传多种格式的文档作为知识库，用户可以通过自然语言对话方式查询文档内容，系统具备对话记忆能力，能够理解上下文关联的问题。

## 二、技术架构

### 2.1 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 前端界面 | Gradio 6.x | Web UI 框架 |
| 后端框架 | LangChain 1.x | LLM 应用开发框架 |
| 向量数据库 | ChromaDB | 本地向量存储 |
| Embedding模型 | 千问 text-embedding-v1 | 文本向量化 |
| 对话模型 | MiniMax MiniMax-M2.7 | 大语言模型 |

### 2.2 双模型架构

本系统采用**双模型分离架构**，将对话能力与向量检索能力解耦：

| 用途 | 模型 | 配置变量 |
|------|------|----------|
| LLM 对话 | MiniMax MiniMax-M2.7 | `MINIMAX_API_KEY`, `MINIMAX_API_BASE`, `MINIMAX_MODEL` |
| 向量 Embedding | 千问 text-embedding-v1 | `QWEN_API_KEY`, `QWEN_API_BASE`, `QWEN_EMBEDDING_MODEL` |

### 2.3 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端层 (app.py)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  对话界面     │  │  知识库管理   │  │  流式输出渲染         │   │
│  │  Chatbot UI  │  │  File Upload │  │  Streaming Display   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
└─────────┼─────────────────┼──────────────────────┼───────────────┘
          │                 │                      │
          ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      业务逻辑层 (rag_engine.py)                   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    RAGEngine 核心引擎                     │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │   │
│  │  │ 文档处理   │  │ 向量检索   │  │ 对话生成           │  │   │
│  │  │ Document   │  │ Retrieval  │  │ LLM Generation     │  │   │
│  │  │ Processing │  │ Engine     │  │ (Stream/Non-Stream)│  │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘  │   │
│  │        │               │                    │             │   │
│  │  ┌─────┴──────┐  ┌─────┴──────┐  ┌─────────┴──────────┐  │   │
│  │  │ 文本分割   │  │ 相似度搜索 │  │ 对话历史管理       │  │   │
│  │  │ TextSplit  │  │ VectorSearch│ │ ChatHistory        │  │   │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据层                                    │
│  ┌────────────────────┐         ┌────────────────────────────┐  │
│  │   ChromaDB         │         │   对话历史 (InMemory)      │  │
│  │   向量数据库       │         │   HumanMessage/AIMessage   │  │
│  │   (持久化存储)     │         │   (内存存储)               │  │
│  └────────────────────┘         └────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      外部服务层                                  │
│  ┌────────────────────────┐    ┌────────────────────────────┐   │
│  │  千问 Embedding API     │    │  MiniMax 对话 API          │   │
│  │  text-embedding-v1     │    │  MiniMax-M2.7              │   │
│  │  向量化存储             │    │  流式文本生成              │   │
│  └────────────────────────┘    └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 核心流程

**文档上传流程**：
```
用户上传文档 → 文档加载(Load) → 文本分割(Split) → 向量化(千问Embedding) → 存储到ChromaDB
```

**问答流程**：
```
用户提问 → 问题向量化(千问) → 检索相似文档 → 组装上下文 → LLM生成回答(MiniMax) → 流式返回 → 更新对话历史
```

## 三、功能特性

### 3.1 文档管理
- **支持格式**：PDF、DOCX、TXT
- **批量上传**：支持一次上传多个文档
- **自动分割**：文档自动分割为文本块（1000字符/块，重叠200字符）
- **向量化存储**：文本块自动向量化并存储到 ChromaDB

### 3.2 智能问答
- **语义检索**：基于向量相似度检索相关文档内容
- **RAG增强**：结合检索到的上下文生成回答
- **对话记忆**：保留完整对话历史，支持上下文关联问答
- **智能回复**：当文档中无相关信息时，基于通用知识回答并说明来源

### 3.3 系统管理
- **清空对话**：清除当前对话历史
- **清空知识库**：删除所有已上传文档及向量数据

## 四、环境配置

### 4.1 前置要求
- Python 3.11+
- **千问 API 密钥**（用于 Embedding 向量化）
- **MiniMax API 密钥**（用于对话生成）

### 4.2 安装依赖
```bash
pip install -r requirements.txt
```

### 4.3 配置参数

编辑 `config.py` 文件：

```python
# ========== MiniMax (LLM 对话) ==========
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "your-minimax-api-key")
MINIMAX_API_BASE = os.getenv("MINIMAX_API_BASE", "https://api.minimaxi.com/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

# ========== 千问 (向量 Embedding) ==========
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "your-qwen-api-key")
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v1")

# ========== 向量数据库配置 ==========
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### 4.4 环境变量配置（可选）

可通过环境变量覆盖上述配置：

```bash
export MINIMAX_API_KEY="your-minimax-api-key"
export QWEN_API_KEY="your-qwen-api-key"
```

## 五、使用说明

### 5.1 启动系统
```bash
python app.py
```
启动后访问：http://localhost:7860

### 5.2 上传文档
1. 切换到「知识库管理」标签页
2. 点击「上传文档」区域选择文件
3. 点击「上传并处理文档」按钮
4. 等待处理完成，查看上传结果提示

### 5.3 进行问答
1. 切换到「对话」标签页
2. 在输入框中输入问题
3. 按回车键或点击「发送」按钮
4. 查看系统回复

### 5.4 清空操作
- **清空对话记忆**：点击对话页面的「清空对话记忆」按钮
- **清空知识库**：在知识库管理页面点击「清空知识库」按钮

## 六、项目结构

```
myQueryKn/
├── config.py              # 配置文件（双模型配置）
├── rag_engine.py          # RAG核心引擎
├── app.py                 # Gradio前端界面
├── demo.py                # API测试脚本
├── requirements.txt       # Python依赖
├── 使用说明.md            # 本文档
└── chroma_db/            # 向量数据库存储目录（自动生成）
```

## 七、核心模块说明

### 7.1 RAGEngine 类

| 方法 | 说明 |
|------|------|
| `add_documents(file_paths)` | 添加文档到知识库 |
| `query(question)` | 根据问题检索并生成回答 |
| `query_stream(question)` | 流式版本，用于实时显示回答 |
| `clear_memory()` | 清空对话历史 |
| `clear_vectorstore()` | 清空知识库 |

### 7.2 QwenEmbeddings 类

自定义的千问 Embedding 适配器，实现 `embed_documents` 和 `embed_query` 两个方法，兼容 LangChain 接口。

## 八、常见问题

### Q1: 启动时报端口被占用
**解决**：修改 `app.py` 中的 `server_port` 参数为其他端口

### Q2: 报 embedding 维度不匹配错误
**解决**：更换 Embedding 模型后需要删除 `chroma_db` 目录重新创建，因为不同模型的向量维度不同

### Q3: 回答不准确
**解决**：
- 检查文档是否成功上传
- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 参数
- 尝试使用更强的对话模型

### Q4: 如何更换模型
**解决**：修改 `config.py` 中的对应配置项

## 九、API 密钥获取

### MiniMax（对话模型）
1. 访问 https://platform.minimax.chat/
2. 注册/登录 MiniMax 账号
3. 开通 API 服务
4. 创建并复制 API 密钥

### 千问（Embedding 模型）
1. 访问 https://dashscope.console.aliyun.com/
2. 注册/登录阿里云账号
3. 开通 DashScope 服务
4. 创建并复制 API 密钥

## 十、双模型配置说明

本系统采用**双 key、双模型、双 baseurl** 架构：

| 服务 | API Key | Base URL | 模型 |
|------|---------|----------|------|
| MiniMax 对话 | `MINIMAX_API_KEY` | `https://api.minimaxi.com/v1` | `MiniMax-M2.7` |
| 千问 Embedding | `QWEN_API_KEY` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v1` |

> 注意：更换 Embedding 模型后（如从 `embo-01` 改为 `text-embedding-v1`），由于向量维度不同，需要删除 `chroma_db` 目录后重新上传文档。