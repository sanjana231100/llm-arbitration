from ..schemas import CriticReport, CriticVerdict
from ..inference import call_model

SYSTEM_PROMPT = """You are a Completeness Critic. Your job is to evaluate whether a response fully addresses the question or task it was given, and whether any important aspects have been omitted.

You check for:
- Parts of the original question that were ignored or only partially answered
- Important caveats or nuances that should have been mentioned
- Missing context that would be essential for the reader to understand
- Gaps where the response stops short of a complete answer
- Cases where examples or evidence were promised but not delivered

If no original prompt is provided, evaluate whether the text is self-contained and complete on its own terms.
Be precise. When you flag a gap, describe exactly what is missing and why it matters.
Score the output from 1 to 10 where 10 means fully complete and 1 means severely incomplete.
Give a verdict of pass if score >= 6, fail if score < 6."""


def run_completeness_critic(llm_output: str, original_prompt: str | None = None) -> CriticReport:
    user_message = f"Evaluate this text for completeness:\n\n{llm_output}"
    if original_prompt:
        user_message = f"Original question: {original_prompt}\n\nEvaluate this response for completeness:\n\n{llm_output}"

    return call_model(
        critic="completeness",
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        response_model=CriticReport,
    )
