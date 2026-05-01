import sys
import json
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio

API_BASE = "http://localhost:8000"

app = Server("llm-arbitration")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="arbitrate_text",
            description=(
                "Evaluates any LLM-generated text using a multi-agent arbitration pipeline. "
                "Routes the text to three specialist critic agents (Factual Accuracy, Logical Consistency, Completeness) "
                "running on different model architectures in parallel, detects disagreements between critics, "
                "then synthesizes a confidence-scored verdict with confirmed issues, dismissed flags, and a summary. "
                "Use this when you want to evaluate the quality, accuracy, or completeness of any AI-generated response."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "llm_output": {
                        "type": "string",
                        "description": "The LLM-generated text to evaluate.",
                    },
                    "original_prompt": {
                        "type": "string",
                        "description": "The original prompt or question that generated this output (optional but improves evaluation quality).",
                    },
                },
                "required": ["llm_output"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "arbitrate_text":
        raise ValueError(f"Unknown tool: {name}")

    llm_output = arguments.get("llm_output", "")
    original_prompt = arguments.get("original_prompt")

    if not llm_output.strip():
        return [TextContent(type="text", text="Error: llm_output cannot be empty.")]

    payload = {"llm_output": llm_output}
    if original_prompt:
        payload["original_prompt"] = original_prompt

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{API_BASE}/v1/arbitrate", json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.ConnectError:
        return [TextContent(
            type="text",
            text="Error: Could not connect to the arbitration API at http://localhost:8000. Make sure the FastAPI server is running with: uvicorn api.main:app --port 8000",
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error calling arbitration API: {e}")]

    verdict = result["verdict"]
    reports = result["critic_reports"]
    disagreements = result["disagreements"]

    lines = [
        f"ARBITRATION VERDICT",
        f"ID: {result['arbitration_id']}",
        f"",
        f"Overall score: {verdict['overall_score']}/10",
        f"Confidence: {verdict['confidence'].upper()}",
        f"",
        f"SUMMARY",
        verdict["summary"],
        f"",
    ]

    if verdict["confirmed_issues"]:
        lines.append("CONFIRMED ISSUES")
        for i, issue in enumerate(verdict["confirmed_issues"], 1):
            lines.append(f"{i}. [{issue['severity'].upper()}] {issue['issue']}")
            lines.append(f'   Quote: "{issue["quote"]}"')
            lines.append(f"   Evidence: {issue['evidence']}")
            lines.append(f"   Flagged by: {', '.join(issue['flagged_by'])}")
            lines.append("")

    if verdict["dismissed_flags"]:
        lines.append("DISMISSED FLAGS")
        for flag in verdict["dismissed_flags"]:
            lines.append(f"- {flag['flagged_by']} flagged: \"{flag['quote']}\"")
            lines.append(f"  Dismissed: {flag['dismissal_reason']}")
            lines.append("")

    if disagreements:
        lines.append("CRITIC DISAGREEMENTS")
        for d in disagreements:
            lines.append(f"- {d['description']}")
        lines.append("")

    lines.append("CRITIC SCORES")
    for report in reports:
        lines.append(
            f"- {report['critic_name']}: {report['score']}/10 ({report['verdict'].upper()})"
        )

    return [TextContent(type="text", text="\n".join(lines))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
