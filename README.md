# 金融年报 RAG 问答系统

本项目基于 RAG Challenge 开源项目进行二次开发与工程化扩展，原项目地址：

[https://github.com/IlyaRice/RAG-Challenge-2/tree/main](https://github.com/IlyaRice/RAG-Challenge-2/tree/main)

项目面向金融年报 PDF 的结构化理解、检索增强问答与跨公司指标比较。整体流程以 Docling 为基础，构建了从 PDF 解析到问答服务的端到端流水线，可用于年报、财报、ESG 报告等长文档场景下的证据型问答。

## 主要工作

- 基于 Docling 构建金融年报 PDF 理解流水线，完成 OCR、版面解析、表格结构化与页级文本重构。
- 通过缺页占位保持文本序列与物理页码一致，支持结构化问答、证据页码回溯及跨公司指标比较。
- 设计实体路由与报告级 RAG 检索架构，为不同年报独立构建 FAISS 索引，提升多公司、多报告场景下的检索隔离性与可扩展性。
- 结合父文档召回与 LLM 重排序生成证据上下文，提高答案生成时的上下文质量与可解释性。
- 基于 FastAPI、Redis 和 Streamlit 扩展异步多轮问答服务，实现任务队列、会话状态隔离、历史记忆与前端交互。
- 在评测流程中将得分由 `101.3` 提升至 `120.7`。

## 系统结构

离线知识构建流程主要包括 PDF 解析、表格序列化、页级 Markdown 重构、文本切块与 FAISS 向量索引构建。相关代码集中在 `src/pipeline.py`、`src/pdf_parsing.py`、`src/tables_serialization.py` 与 `src/retrieval.py`。

在线问答流程包括实体识别、报告路由、候选证据召回、LLM 重排序与结构化答案生成。相关代码集中在 `src/questions_processing.py`、`src/routing.py`、`src/reranking.py`、`src/api_requests.py` 与 `src/prompts.py`。

服务化部分提供同步和异步两套使用方式：

- `api_single.py` / `app_single.py`：同步 API 与 Streamlit 单轮交互。
- `api.py` / `worker.py` / `app.py`：基于 Redis 队列的异步问答服务。

## 快速开始

```bash
git clone https://github.com/zzqie/RAG-Challenge.git
cd RAG-Challenge
python -m venv venv
venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt
```

将 `env` 重命名为 `.env`，并根据需要配置模型或 API Key。

## 数据与评测

仓库中保留了 RAG Challenge 的测试数据、PDF 报告、解析结果、向量库与多轮实验输出，便于复现检索、问答和评分流程。`round2/` 目录包含第二轮评测相关答案与排名文件。

## License

MIT
