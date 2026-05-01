import wandb
from .schemas import ArbitrateResponse
from .config import get_settings


def init_wandb() -> bool:
    settings = get_settings()
    if not settings.wandb_api_key:
        return False
    try:
        wandb.init(
            project="llm-arbitration",
            reinit=True,
            mode="online",
        )
        return True
    except Exception as e:
        print(f"W&B init failed: {e}")
        return False


def log_arbitration(response: ArbitrateResponse) -> None:
    settings = get_settings()
    if not settings.wandb_api_key:
        return

    try:
        if wandb.run is None:
            wandb.init(project="llm-arbitration", reinit=True, mode="online")

        verdict = response.verdict
        reports = response.critic_reports

        scores = {r.critic_name: r.score for r in reports}
        verdicts = {r.critic_name: r.verdict.value for r in reports}
        flag_counts = {r.critic_name: len(r.flags) for r in reports}

        wandb.log({
            "overall_score": verdict.overall_score,
            "confidence": verdict.confidence.value,
            "confirmed_issue_count": len(verdict.confirmed_issues),
            "dismissed_flag_count": len(verdict.dismissed_flags),
            "disagreement_count": len(response.disagreements),

            "factual_score": scores.get("Factual Accuracy Critic", 0),
            "logical_score": scores.get("Logical Consistency Critic", 0),
            "completeness_score": scores.get("Completeness Critic", 0),

            "factual_verdict": verdicts.get("Factual Accuracy Critic", ""),
            "logical_verdict": verdicts.get("Logical Consistency Critic", ""),
            "completeness_verdict": verdicts.get("Completeness Critic", ""),

            "factual_flag_count": flag_counts.get("Factual Accuracy Critic", 0),
            "logical_flag_count": flag_counts.get("Logical Consistency Critic", 0),
            "completeness_flag_count": flag_counts.get("Completeness Critic", 0),

            "critics_all_agree": len(response.disagreements) == 0,
            "any_critic_overruled": len(verdict.dismissed_flags) > 0,
        })

    except Exception as e:
        print(f"W&B logging failed (non-fatal): {e}")
