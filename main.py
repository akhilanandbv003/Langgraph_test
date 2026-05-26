from dotenv import load_dotenv
load_dotenv()

import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import app


def build_initial_state(query: str, prior_messages: list = None) -> dict:
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


def run():
    prior_messages = []
    pending_query = None

    try:
        while True:
            query = pending_query or input("> ").strip()
            pending_query = None
            if not query:
                continue

            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            state = build_initial_state(query, prior_messages)
            result = app.invoke(state, config)

            # HITL: graph paused inside clarity_agent waiting for clarification
            if app.get_state(config).next:
                print("Could you clarify your question? Please mention the company name.")
                clarification = input("> ").strip()
                if not clarification:
                    continue
                result = app.invoke(Command(resume=clarification), config)

            print(f"\n{result['final_response']}\n")
            prior_messages = result["messages"]

            follow_up = input("Follow-up question? (or 'quit'): ").strip()
            if follow_up.lower() == "quit":
                break
            if follow_up:
                pending_query = follow_up

    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    run()
