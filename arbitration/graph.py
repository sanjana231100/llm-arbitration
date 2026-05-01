import uuid
from langgraph.graph import StateGraph, END
from .state import ArbitrationState
from .schemas import CriticReport, Disagreement
from .critics.factual import run_factual_critic
from .critics.logical import run_logical_critic
from .critics.completeness import run_completeness_critic
from .adjudicator import run_adjudicator
from .config import get_settings


def _run_factual(state: ArbitrationState) -> dict:
    report = run_factual_critic(
        state["llm_output"],
        state.get("original_prompt"),
    )
    report.critic_name = "Factual Accuracy Critic"
    return {"factual_report": report}


def _run_logical(state: ArbitrationState) -> dict:
    report = run_logical_critic(
        state["llm_output"],
        state.get("original_prompt"),
    )
    report.critic_name = "Logical Consistency Critic"
    return {"logical_report": report}


def _run_completeness(state: ArbitrationState) -> dict:
    report = run_completeness_critic(
        state["llm_output"],
        state.get("original_prompt"),
    )
    report.critic_name = "Completeness Critic"
    return {"completeness_report": report}


def _detect_disagreements(state: ArbitrationState) -> dict:
    settings = get_settings()
    threshold = settings.disagreement_threshold

    reports: list[tuple[str, CriticReport]] = [
        ("Factual Accuracy Critic", state["factual_report"]),
        ("Logical Consistency Critic", state["logical_report"]),
        ("Completeness Critic", state["completeness_report"]),
    ]

    disagreements: list[Disagreement] = []

    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            name_a, report_a = reports[i]
            name_b, report_b = reports[j]
            delta = abs(report_a.score - report_b.score)

            if delta >= threshold:
                disagreements.append(
                    Disagreement(
                        critic_a=name_a,
                        critic_b=name_b,
                        score_a=report_a.score,
                        score_b=report_b.score,
                        delta=delta,
                        description=(
                            f"{name_a} scored {report_a.score} but "
                            f"{name_b} scored {report_b.score} "
                            f"(delta={delta})"
                        ),
                    )
                )

    return {
        "disagreements": disagreements,
        "arbitration_id": str(uuid.uuid4()),
    }


def _run_adjudicator(state: ArbitrationState) -> dict:
    verdict = run_adjudicator(
        llm_output=state["llm_output"],
        factual_report=state["factual_report"],
        logical_report=state["logical_report"],
        completeness_report=state["completeness_report"],
        disagreements=state["disagreements"],
        original_prompt=state.get("original_prompt"),
    )
    return {"verdict": verdict}


def build_graph():
    graph = StateGraph(ArbitrationState)

    graph.add_node("factual_critic", _run_factual)
    graph.add_node("logical_critic", _run_logical)
    graph.add_node("completeness_critic", _run_completeness)
    graph.add_node("detect_disagreements", _detect_disagreements)
    graph.add_node("adjudicator", _run_adjudicator)

    graph.add_edge("__start__", "factual_critic")
    graph.add_edge("__start__", "logical_critic")
    graph.add_edge("__start__", "completeness_critic")

    graph.add_edge("factual_critic", "detect_disagreements")
    graph.add_edge("logical_critic", "detect_disagreements")
    graph.add_edge("completeness_critic", "detect_disagreements")

    graph.add_edge("detect_disagreements", "adjudicator")
    graph.add_edge("adjudicator", END)

    return graph.compile()


arbitration_graph = build_graph()
