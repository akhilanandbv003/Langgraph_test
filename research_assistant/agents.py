import json
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import interrupt

from state import ResearchState
from research_assistant.tools import search_company
from gateway import llm_gateway


def clarity_agent(state: ResearchState) -> dict:
    print("[agent] clarity")
    system = SystemMessage(content=(
        "You are a query clarity checker. The user wants to research a company. "
        "Determine whether the company name in their query is clear and specific enough to research. "
        "Reply with exactly one word: 'clear' if the company is identifiable, "
        "or 'needs_clarification' if the query is ambiguous, generic, or missing a company name."
    ))
    history = state["messages"]
    user_msg = HumanMessage(content=f"Query: {state['query']}")
    raw = llm_gateway([system] + history + [user_msg]).strip().lower()
    status = "clear" if "clear" in raw and "needs_clarification" not in raw else "needs_clarification"

    if status == "needs_clarification":
        clarification = interrupt("Which company are you asking about? Please be specific.")
        return {"query": clarification, "clarity_status": "clear"}

    return {"clarity_status": "clear"}


def research_agent(state: ResearchState) -> dict:
    print(f"[agent] research  (attempt {state['attempts'] + 1})")
    system = SystemMessage(content=(
        "You are a financial research analyst. You will be given raw search results about a company. "
        "Analyze them and respond with a JSON object containing exactly two keys:\n"
        '  "findings": a concise summary of the key information (string)\n'
        '  "confidence_score": an integer from 0 to 10 reflecting how complete and reliable the findings are\n'
        "Respond with JSON only — no markdown fences, no extra text."
    ))
    raw_results = search_company(state["query"])
    history = state["messages"]
    user_msg = HumanMessage(content=(
        f"Company query: {state['query']}\n\nSearch results:\n{raw_results}"
    ))
    response = llm_gateway([system] + history + [user_msg])

    try:
        parsed = json.loads(response.strip())
        findings = str(parsed.get("findings", response.strip()))
        confidence_score = int(parsed.get("confidence_score", 5))
    except (json.JSONDecodeError, ValueError):
        findings = response.strip()
        confidence_score = 5

    return {
        "research_findings": findings,
        "confidence_score": confidence_score,
        "attempts": state["attempts"] + 1,
    }


def validator_agent(state: ResearchState) -> dict:
    print("[agent] validator")
    system = SystemMessage(content=(
        "You are a research quality validator. Given an original query and research findings, "
        "decide whether the findings sufficiently answer the query. "
        "Reply with exactly one word: 'sufficient' if the findings adequately address the query, "
        "or 'insufficient' if key information is missing or the findings are too vague."
    ))
    user_msg = HumanMessage(content=(
        f"Original query: {state['query']}\n\nFindings:\n{state['research_findings']}"
    ))
    raw = llm_gateway([system, user_msg]).strip().lower()
    result = "sufficient" if "sufficient" in raw and "insufficient" not in raw else "insufficient"
    return {"validation_result": result}


def synthesis_agent(state: ResearchState) -> dict:
    print("[agent] synthesis")
    system = SystemMessage(content=(
        "You are a professional research summarizer. Using the conversation history and research findings, "
        "write a clear, concise, user-friendly summary that directly answers the user's original question. "
        "Use plain language. Avoid jargon. Structure the response with a brief intro, key findings, "
        "and a closing takeaway."
    ))
    history = state["messages"]
    user_msg = HumanMessage(content=(
        f"Original query: {state['query']}\n\nResearch findings:\n{state['research_findings']}"
    ))
    summary = llm_gateway([system] + history + [user_msg]).strip()
    return {
        "final_response": summary,
        "messages": [AIMessage(content=summary)],
    }
