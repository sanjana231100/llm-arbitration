# LLM Arbitration System

A multi-agent system that takes any LLM-generated output, routes it to three specialized critic agents running on different model architectures, detects disagreements, and synthesizes a confidence-scored verdict.

## Stack

- **LangGraph** — parallel critic fan-out + disagreement detection
- **Groq API** — Llama 3.3 70b, Mixtral 8x7b, Gemma2 9b (one per critic)
- **instructor + Pydantic** — structured outputs enforced on every model call
- **FastAPI** — REST API backend with OpenAPI docs
- **Streamlit** — verdict explorer UI with inline annotations
- **MCP server** — exposes arbitration as a native tool for Claude Desktop / Cursor
- **Ollama** — local inference fallback
- **W&B** — critic behavior analytics
- **Docker + Docker Compose** — containerized deployment

## Setup

```bash
git clone https://github.com/sanjana231100/llm-arbitration
cd llm-arbitration

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# fill in your API keys
```

## Run

```bash
# FastAPI backend
uvicorn api.main:app --reload --port 8000

# Streamlit UI (separate terminal)
streamlit run ui/app.py

# MCP server (separate terminal)
python mcp_server/server.py
```

## API

Once the backend is running, visit `http://localhost:8000/docs` for the interactive OpenAPI documentation.

## Agents

| Agent | Model | Dimension |
|---|---|---|
| Factual Accuracy Critic | Llama 3.3 70b (Groq) | Verifiable claims |
| Logical Consistency Critic | Mixtral 8x7b (Groq) | Reasoning validity |
| Completeness Critic | Gemma2 9b (Groq) | Coverage gaps |
| Adjudicator | Llama 3.3 70b (Groq) | Conflict resolution |
