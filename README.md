---
title: LLM Arbitration System
emoji: ⚖
colorFrom: purple
colorTo: blue
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
- **Streamlit** — verdict explorer
git add README.md
git commit -m "fix: HuggingFace valid colorTo value"
git push origin main
git push hf main --force
