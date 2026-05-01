from ..schemas import CriticReport, CriticVerdict
from ..inference import call_model

SYSTEM_PROMPT = """You are a Factual Accuracy Critic. Your job is to evaluate whether the claims in a given text are accurate, verifiable, and internally consistent.

You check for:
- Statements that contradict established facts
- Numbers, dates, or statistics that appear incorrect
- Claims that are internally inconsistent with other parts of the text
- Assertions presented as fact that are actually speculation or opinion

Be precise. When you flag an issue, quote the exact text and explain what is wrong with it.
Score the output from 1 to 10 where 10 means fully accurate and 1 means severely inaccurate.
Give a verdict of pass if score >= 6, fail if score < 6."""


def run_factual_critic(llm_output: str, original_prompt: str | None = None) -> CriticReport:
    user_message = f"Evaluate this text for factual accuracy:\n\n{llm_output}"
    if original_prompt:
        user_message = f"Original question: {original_prompt}\n\nEvaluate this response for factual accuracy:\n\n{llm_output}"

    return call_model(
        critic="factual",
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        response_model=CriticReport,
    )
