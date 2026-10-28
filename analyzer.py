"""Public entry point used by both the CLI and the Streamlit UI."""
from dataclasses import dataclass
from typing import Optional

from app.graph import build_graph
from app.llm import ConfigError, get_llm
from app.schemas import HiringReport
from app.utils import validate_input


@dataclass
class AnalysisResult:
    report: Optional[HiringReport] = None
    error: Optional[str] = None
    state: Optional[dict] = None  # full final state (useful to show each node's output)

    @property
    def ok(self) -> bool:
        return self.report is not None and self.error is None


def analyze_job_description(text, provider=None, api_key=None, llm=None) -> AnalysisResult:
    """Validate input, run the LangGraph pipeline, and never raise to the caller."""
    try:
        cleaned = validate_input(text)
    except ValueError as exc:
        return AnalysisResult(error=str(exc))

    try:
        llm = llm or get_llm(provider, api_key)
    except ConfigError as exc:
        return AnalysisResult(error=str(exc))

    try:
        graph = build_graph(llm)
        state = graph.invoke({"job_description": cleaned})
    except Exception as exc:
        return AnalysisResult(error=f"Unexpected error while analyzing: {exc}")

    if state.get("error"):
        return AnalysisResult(error=state["error"], state=state)
    return AnalysisResult(report=state["report"], state=state)
