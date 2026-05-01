---
title: LLM Arbitration System
emoji: ⚖
colorFrom: purple
colorTo: teal
sdk: docker
pinned: false
app_port: 7860
---

# LLM Arbitration System

A multi-agent system that takes any LLM-generated output, routes it to three specialized critic agents running on different model architectures, detects disagreements, and synthesizes a confidence-scored verdict.

## Stack

- **LangGraph** — parallel critic fan-out + disagreement detection
- **Groq API** — Llama 3.3 70b, Qwen3 32b, Llama 4 Scout (one per critic)
- **instructor + Pydantic** — structured outputs enforced on every model call
- **FastAPI** — REST API backend with OpenAPI docs at /docs
- **Streamlit** — verdict explorer UI with inline annotations
- **MCP server** — exposes arbitration as a native tool for Claude Desktop / Cursor
- **Ollama** — local inference fallback
- **W&B** — critic behavior analytics
- **Docker + Docker Compose** — containerized deployment

## Agents

| Agent | Model | Dimension |
|---|---|---|
| Factual Accuracy Critic | Llama 3.3 70b (Groq) | Verifiable claims |
| Logical Consistency Critic | Qwen3 32b (Groq) | Reasoning validity |
| Completeness Critic | Llama 4 Scout (Groq) | Coverage gaps |
| Adjudicator | Llama 3.3 70b (Groq) | Conflict resolution |

## Local setup

```bash
git clone https://github.com/sanjana231100/llm-arbitration
cd llm-arbitration
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your API keys to .env

uvicorn api.main:app --reload --port 8000
streamlit run ui/app.py
python mcp_server/server.py
```

## API

Visit `http://localhost:8000/docs` for interactive OpenAPI documentation.
