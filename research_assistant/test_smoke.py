from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from graph import app


def make_state(query: str, prior_messages: list = None) -> dict:
    messages = (prior_messages or []) + [HumanMessage(content=query)]
    return {
        "query": query,
        "messages": messages,
        "clarity_status": "",
        "research_findings": "",
        "confidence_score": 0,
        "attempts": 0,
        "validation_result": "",
        "final_response": "",
    }


# ── Scenario 1: Clear query runs to completion ────────────────────────────────
def test_clear_query():
    query = "What is Apple's latest financial performance?"
    print(f"\n[Scenario 1] Clear query: {query!r}")
    result = app.invoke(make_state(query))
    try:
        assert result["clarity_status"] == "clear", \
            f"Expected 'clear', got {result['clarity_status']!r}"
        assert result["final_response"], "final_response should be non-empty"
        print(f"  clarity_status  : {result['clarity_status']}")
        print(f"  confidence_score: {result['confidence_score']}")
        print(f"  attempts        : {result['attempts']}")
        print(f"  final_response  : {result['final_response'][:120]}...")
        print("PASS")
    except AssertionError as e:
        print(f"FAIL — {e}")


# ── Scenario 2: Ambiguous query stops at clarity ──────────────────────────────
def test_ambiguous_query():
    query = "Tell me about the company"
    print(f"\n[Scenario 2] Ambiguous query: {query!r}")
    result = app.invoke(make_state(query))
    try:
        assert result["clarity_status"] == "needs_clarification", \
            f"Expected 'needs_clarification', got {result['clarity_status']!r}"
        assert result["research_findings"] == "", "No research should run on ambiguous query"
        assert result["attempts"] == 0, "No research attempts should be made"
        print(f"  clarity_status   : {result['clarity_status']}")
        print(f"  research_findings: {result['research_findings']!r}")
        print(f"  attempts         : {result['attempts']}")
        print("PASS")
    except AssertionError as e:
        print(f"FAIL — {e}")


# ── Scenario 3: Clarification loop — ambiguous → clarify → research ───────────
def test_clarification_then_research():
    """Spec: 'Resume processing after receiving clarification'"""
    ambiguous = "Tell me about the company"
    print(f"\n[Scenario 3] Clarification loop: {ambiguous!r}")
    first = app.invoke(make_state(ambiguous))
    try:
        assert first["clarity_status"] == "needs_clarification"

        # simulate user providing clarification
        clarification = "I meant Tesla — what are their latest financials?"
        updated_messages = first["messages"] + [HumanMessage(content=clarification)]
        second = app.invoke({
            **first,
            "query": clarification,
            "messages": updated_messages,
            "clarity_status": "",
            "attempts": 0,
        })

        assert second["clarity_status"] == "clear", \
            f"Expected 'clear' after clarification, got {second['clarity_status']!r}"
        assert second["final_response"], "Should have a final response after clarification"
        print(f"  clarity_status after clarification: {second['clarity_status']}")
        print(f"  final_response: {second['final_response'][:120]}...")
        print("PASS")
    except AssertionError as e:
        print(f"FAIL — {e}")


# ── Scenario 4: Multi-turn follow-up carries conversation history ─────────────
def test_multi_turn_followup():
    """Spec: 'Support follow-up questions like What about their competitors?'"""
    query1 = "What is Microsoft's latest financial performance?"
    print(f"\n[Scenario 4] Multi-turn follow-up")
    result1 = app.invoke(make_state(query1))
    try:
        assert result1["clarity_status"] == "clear"
        assert result1["final_response"]

        # follow-up carries prior messages
        query2 = "What about their main competitors?"
        result2 = app.invoke(make_state(query2, prior_messages=result1["messages"]))

        assert result2["clarity_status"] == "clear", \
            f"Follow-up should be clear, got {result2['clarity_status']!r}"
        assert result2["final_response"], "Follow-up should produce a response"
        # history should include messages from both turns
        assert len(result2["messages"]) > len(result1["messages"]), \
            "Message history should grow across turns"
        print(f"  turn 1 messages : {len(result1['messages'])}")
        print(f"  turn 2 messages : {len(result2['messages'])}")
        print(f"  follow-up response: {result2['final_response'][:120]}...")
        print("PASS")
    except AssertionError as e:
        print(f"FAIL — {e}")


# ── Scenario 5: confidence_score is within valid range ───────────────────────
def test_confidence_score_range():
    """Spec: 'assigns a confidence_score (0-10)'"""
    query = "What is Google's latest financial performance?"
    print(f"\n[Scenario 5] Confidence score range: {query!r}")
    result = app.invoke(make_state(query))
    try:
        score = result["confidence_score"]
        assert isinstance(score, int), f"confidence_score must be int, got {type(score)}"
        assert 0 <= score <= 10, f"confidence_score {score} is outside 0–10"
        print(f"  confidence_score: {score} (valid range 0–10)")
        print("PASS")
    except AssertionError as e:
        print(f"FAIL — {e}")


# ── Scenario 6: All required state fields are populated after a clear query ───
def test_state_fields_populated():
    """Spec: 'Use a proper state schema that includes...'"""
    query = "What is Amazon's business strategy?"
    print(f"\n[Scenario 6] State fields populated: {query!r}")
    result = app.invoke(make_state(query))
    try:
        assert result["clarity_status"] == "clear"
        assert result["research_findings"], "research_findings must be set"
        assert result["validation_result"] in ("sufficient", "insufficient", ""), \
            f"Unexpected validation_result: {result['validation_result']!r}"
        assert result["attempts"] >= 1, "At least one research attempt must be made"
        assert result["final_response"], "final_response must be set"
        assert len(result["messages"]) >= 1, "messages must be non-empty"
        print(f"  research_findings : {result['research_findings'][:80]}...")
        print(f"  validation_result : {result['validation_result']!r}")
        print(f"  attempts          : {result['attempts']}")
        print(f"  messages count    : {len(result['messages'])}")
        print("PASS")
    except AssertionError as e:
        print(f"FAIL — {e}")


# ── Scenario 7: Max attempts guard — graph terminates within 3 retries ────────
def test_max_attempts_guard():
    """Spec: 'Research Agent loop back if insufficient AND attempts < 3'"""
    query = "What is Netflix's content strategy?"
    print(f"\n[Scenario 7] Max attempts guard: {query!r}")
    result = app.invoke(make_state(query))
    try:
        assert result["attempts"] <= 3, \
            f"attempts {result['attempts']} exceeded max of 3"
        assert result["final_response"], "Should always reach synthesis within 3 attempts"
        print(f"  attempts      : {result['attempts']} (max allowed: 3)")
        print(f"  final_response: {result['final_response'][:120]}...")
        print("PASS")
    except AssertionError as e:
        print(f"FAIL — {e}")


if __name__ == "__main__":
    test_clear_query()
    test_ambiguous_query()
    test_clarification_then_research()
    test_multi_turn_followup()
    test_confidence_score_range()
    test_state_fields_populated()
    test_max_attempts_guard()
