from .schemas import (
    CriticReport,
    Disagreement,
    ArbitrationVerdict,
    ConfirmedIssue,
    DismissedFlag,
    ConfidenceLevel,
)
from .inference import call_model
from pydantic import BaseModel, Field


class AdjudicatorInput(BaseModel):
    overall_score: int = Field(ge=1, le=10)
    confidence: ConfidenceLevel
    confirmed_issues: list[ConfirmedIssue] = Field(default_factory=list)
    dismissed_flags: list[DismissedFlag] = Field(default_factory=list)
    summary: str


SYSTEM_PROMPT = """You are an Adjudicator agent. You receive reports from three specialist critic agents who have independently evaluated an LLM-generated text:

1. Factual Accuracy Critic — checks whether claims are correct and verifiable
2. Logical Consistency Critic — checks whether reasoning is valid and conclusions follow
3. Completeness Critic — checks whether the response fully addresses the question

Your job is to:
- Weigh all three reports and resolve any disagreements between critics
- Decide which flagged issues are genuinely confirmed and which should be dismissed
- Produce a single overall quality score from 1 to 10
- Set a confidence level: high if critics mostly agree, medium if there are some disagreements, low if critics strongly disagree
- List confirmed issues with the exact quote, severity, evidence, and which critics flagged it
- List dismissed flags with your reasoning for overruling the critic
- Write one clear summary paragraph of your overall assessment

When critics disagree, use your judgment about which critic's reasoning is stronger. A factual error flagged by the factual critic should be taken seriously even if other critics scored the text highly. Logical flaws flagged by the logic critic deserve weight even if the text is factually accurate.

Be precise and evidence-based. Do not dismiss flags without a clear reason."""


def run_adjudicator(
    llm_output: str,
    factual_report: CriticReport,
    logical_report: CriticReport,
    completeness_report: CriticReport,
    disagreements: list[Disagreement],
    original_prompt: str | None = None,
) -> ArbitrationVerdict:
    reports_text = f"""
FACTUAL ACCURACY CRITIC (score: {factual_report.score}/10, verdict: {factual_report.verdict}):
Reasoning: {factual_report.reasoning}
Flags:
{chr(10).join(f'- [{f.severity}] "{f.quote}" — {f.issue} (evidence: {f.evidence})' for f in factual_report.flags) or "None"}

LOGICAL CONSISTENCY CRITIC (score: {logical_report.score}/10, verdict: {logical_report.verdict}):
Reasoning: {logical_report.reasoning}
Flags:
{chr(10).join(f'- [{f.severity}] "{f.quote}" — {f.issue} (evidence: {f.evidence})' for f in logical_report.flags) or "None"}

COMPLETENESS CRITIC (score: {completeness_report.score}/10, verdict: {completeness_report.verdict}):
Reasoning: {completeness_report.reasoning}
Flags:
{chr(10).join(f'- [{f.severity}] "{f.quote}" — {f.issue} (evidence: {f.evidence})' for f in completeness_report.flags) or "None"}
"""

    disagreements_text = ""
    if disagreements:
        disagreements_text = "\nDETECTED DISAGREEMENTS:\n"
        disagreements_text += "\n".join(f"- {d.description}" for d in disagreements)
    else:
        disagreements_text = "\nNo significant disagreements detected between critics."

    prompt_context = ""
    if original_prompt:
        prompt_context = f"\nORIGINAL PROMPT: {original_prompt}\n"

    user_message = f"""{prompt_context}
TEXT BEING EVALUATED:
{llm_output}

CRITIC REPORTS:
{reports_text}
{disagreements_text}

Produce your adjudication verdict now."""

    result = call_model(
        critic="adjudicator",
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        response_model=AdjudicatorInput,
    )

    return ArbitrationVerdict(
        overall_score=result.overall_score,
        confidence=result.confidence,
        confirmed_issues=result.confirmed_issues,
        dismissed_flags=result.dismissed_flags,
        summary=result.summary,
    )
