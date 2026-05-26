# LangGraph Multi-Agent Research Assistant

A multi-agent pipeline that researches companies using 4 specialized agents: **Clarity → Research → Validator → Synthesis**.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run the chat

```bash
python main.py
```

Type a company research question at the `>` prompt. If your query is ambiguous, the assistant will pause and ask for clarification before continuing. Type `quit` to exit.

**Example session:**

```
> Tell me about the company
Could you clarify your question? Please mention the company name.
> I meant Nvidia
[agent] clarity
[agent] research  (attempt 1)
[agent] synthesis

Nvidia is a leading semiconductor company...

Follow-up question? (or 'quit'): What about their competitors?
```

## Run the smoke tests

```bash
python -m research_assistant.test_smoke
```

Runs 7 scenarios end-to-end against the live API:

| # | Scenario | What it checks |
|---|---|---|
| 1 | Clear query | `clarity_status == "clear"`, non-empty `final_response` |
| 2 | Ambiguous query | `clarity_status == "needs_clarification"`, no research runs |
| 3 | Clarification loop | Ambiguous → clarify → research completes |
| 4 | Multi-turn follow-up | Conversation history grows across turns |
| 5 | Confidence score range | `0 <= confidence_score <= 10` |
| 6 | State fields populated | All required fields set after a clear query |
| 7 | Max attempts guard | Graph always terminates within 3 research attempts |

## Project structure

```
├── main.py                        # CLI entry point
├── graph.py                       # StateGraph definition
├── state.py                       # ResearchState schema
├── gateway.py                     # LLM abstraction (Anthropic / OpenAI)
└── research_assistant/
    ├── agents.py                  # 4 node functions
    ├── tools.py                   # search_company stub
    └── test_smoke.py              # smoke tests
```

## Environment variables

```
ANTHROPIC_API_KEY=   # required
OPENAI_API_KEY=      # optional, only needed if switching gateway to openai
```
