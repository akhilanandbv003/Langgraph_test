from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import ResearchState
from research_assistant.agents import (
    clarity_agent,
    research_agent,
    validator_agent,
    synthesis_agent,
)

graph = StateGraph(ResearchState)

graph.add_node("clarity", clarity_agent)
graph.add_node("research", research_agent)
graph.add_node("validator", validator_agent)
graph.add_node("synthesis", synthesis_agent)

graph.set_entry_point("clarity")


def route_clarity(state: ResearchState) -> str:
    return "research" if state["clarity_status"] == "clear" else END


def route_research(state: ResearchState) -> str:
    return "synthesis" if state["confidence_score"] >= 6 else "validator"


def route_validator(state: ResearchState) -> str:
    if state["validation_result"] == "insufficient" and state["attempts"] < 3:
        return "research"
    return "synthesis"


graph.add_conditional_edges("clarity", route_clarity)
graph.add_conditional_edges("research", route_research)
graph.add_conditional_edges("validator", route_validator)

graph.add_edge("synthesis", END)

app = graph.compile(checkpointer=MemorySaver())
