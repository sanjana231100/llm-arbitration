from ..schemas import CriticReport, CriticVerdict
from ..inference import call_model

SYSTEM_PROMPT = """You are a Logical Consistency Critic. Your job is to evaluate whether the reasoning in a given text is valid and whether conclusions are properly supported by the evidence or premises given.

You check for:
- Non-sequiturs where conclusions do not follow from the premises
- Circular reasoning or begging the question
- Contradictory statements within the same text
- Logical fallacies such as false dichotomies, straw men, or appeals to authority
- Conclusions that overreach the evidence presented

Be precise. When you flag an issue, quote the exact text and explain the logical flaw.
Score the output from 1 to 10 where 10 means perfectly logical and 1 means severely flawed reasoning.
Give a verdict of pass if score >= 6, fail if score < 6."""


def run_logical_critic(llm_output: str, original_prompt: str | None = None) -> CriticReport:
    user_message = f"Evaluate this text for logical consistency:\n\n{llm_output}"
    if original_prompt:
        user_message = f"Original question: {original_prompt}\n\nEvaluate this response for logical consistency:\n\n{llm_output}"

    return call_model(
        critic="logical",
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        response_model=CriticReport,
    )
