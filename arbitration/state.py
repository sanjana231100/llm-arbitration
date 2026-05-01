from typing import TypedDict, Optional
from .schemas import CriticReport, ArbitrationVerdict, Disagreement


class ArbitrationState(TypedDict):
    llm_output: str
    original_prompt: Optional[str]

    factual_report: Optional[CriticReport]
    logical_report: Optional[CriticReport]
    completeness_report: Optional[CriticReport]

    disagreements: Optional[list[Disagreement]]

    verdict: Optional[ArbitrationVerdict]

    arbitration_id: Optional[str]
