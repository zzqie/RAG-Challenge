# Enterprise Financial Document Intelligence Platform

This repository can be understood as an enterprise-grade financial document intelligence platform built on top of a high-performance RAG pipeline. Its current implementation focuses on annual reports, but the same architecture fits broader business scenarios such as:

- group finance and strategy teams comparing multiple entities and periods
- buy-side and sell-side research workflows
- bank credit review and counterparty due diligence
- IR, ESG, compliance, and audit support
- knowledge extraction from regulated filings and long-form PDF documents

At its core, the system combines:

- custom PDF parsing with Docling
- page reconstruction and markdown normalization
- chunk-level vector retrieval
- optional table serialization for better structured evidence recall
- LLM reranking for context refinement
- structured answer generation with page-level traceability
- multi-entity comparison and conversational memory extensions

## Origin

The project originated from a winning solution in the RAG Challenge competition and was later extended toward a service-oriented financial QA workflow.

Read more about the original background:
- Russian: https://habr.com/ru/articles/893356/
- English: https://abdullin.com/ilya/how-to-build-best-rag/

## Business Framing

Although the current code stores one vector index per report/company file, that design can be interpreted in enterprise terms as a document-sharded retrieval architecture rather than a literal "one-company-one-system" design.

This is often useful in real business environments because it provides:

- natural document and entity isolation
- easier incremental updates when a new filing arrives
- stronger explainability and source attribution
- better control over tenant, entity, or report-scope retrieval
- a straightforward path toward portfolio-level routing before retrieval

In other words, the current implementation already maps well to a multi-entity filing intelligence platform, especially when positioned as:

- report-level knowledge shards
- metadata-driven routing across entities, periods, and document types
- evidence-grounded answering over regulated documents

For a Chinese business-oriented packaging and architecture narrative, see [BUSINESS_PACKAGING_CN.md](BUSINESS_PACKAGING_CN.md).

## Current System Layers

### 1. Offline Knowledge Construction

The pipeline in `src/` builds retrieval-ready knowledge assets from PDF filings:

1. parse PDFs
2. optionally serialize tables with LLMs
3. normalize page content into retrieval-friendly markdown/text
4. split pages into chunks
5. build FAISS vector databases

Main orchestrator: `src/pipeline.py`

### 2. Online Question Answering

The question answering layer:

- identifies target entities from the question
- retrieves relevant chunks or pages
- optionally reranks them with an LLM
- generates structured answers with references
- supports comparative reasoning across multiple companies

Core modules:

- `src/questions_processing.py`
- `src/retrieval.py`
- `src/reranking.py`
- `src/api_requests.py`
- `src/prompts.py`

### 3. Service Layer Extensions

The repository also includes a lightweight service-oriented wrapper for interactive use:

- `api_single.py` / `app_single.py`: synchronous API + Streamlit flow
- `api.py` / `worker.py` / `app.py`: Redis queue based asynchronous flow

This layer adds:

- session history
- structured memory extraction
- cross-entity comparison shortcuts
- UI-facing task orchestration

## Quick Start

```bash
git clone https://github.com/IlyaRice/RAG-Challenge-2.git
cd RAG-Challenge-2
python -m venv venv
venv\Scripts\Activate.ps1
pip install -e . -r requirements.txt
```

Rename `env` to `.env` and add your API keys.

## Datasets

The repository includes:

1. `data/test_set/` - a small local dataset with reports, metadata, processed assets, and sample outputs
2. `data/erc2_set/` - the larger competition dataset description and answer files

These datasets are useful not only for reproducing the original evaluation setup, but also for demonstrating a realistic filing-intelligence workflow end to end.

## Usage

Run the CLI help:

```bash
python main.py --help
```

Available commands:

- `download-models`
- `parse-pdfs`
- `serialize-tables`
- `process-reports`
- `process-questions`

Example:

```bash
cd .\data\test_set\
python ..\..\main.py process-questions --config max_nst_o3m
```

You can also run individual stages directly from `src/pipeline.py` by uncommenting the desired method.

## Recommended Positioning of Existing Configs

- `max_nst_o3m`: balanced high-accuracy filing QA configuration
- `max_st_o3m`: stronger table-aware configuration
- `gemini_thinking`: large-context document reading baseline

See `src/pipeline.py` for the full list of run configurations.

## Practical Caveats

- IBM Watson integration is historical and may not work in a new environment
- the project is strong technically but still research/prototype flavored in engineering style
- tests and production hardening are limited
- PDF parsing benefits significantly from GPU resources

## License

MIT
