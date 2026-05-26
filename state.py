from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages




class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    clarity_status: str           # "clear" | "needs_clarification"
    research_findings: str
    confidence_score: int         # 0–10
    attempts: int
    validation_result: str        # "sufficient" | "insufficient"
    final_response: str
