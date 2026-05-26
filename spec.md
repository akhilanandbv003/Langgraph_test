Problem Statement
Build a multi-agent research assistant using LangGraph that helps users gather information about companies. 
The system should have multiple specialized agents working together, 
support follow-up questions, and request human clarification when queries are ambiguous.

Input: chatbot prompt string:
Output: 

Requirements
1. Agent Architecture
Implement 4 specialized agents:
a) Clarity Agent
Analyzes if the user's query is clear and specific
Checks if company name is mentioned or if query is too vague
OUTPUT: Sets clarity_status to "clear" or "needs_clarification"
ROUTES TO: Interrupt (if unclear) OR Research Agent (if clear)
b) Research Agent
Searches for company information (news, financials, recent developments)
This agent should derive this information from a search tool (Tavily MCP would be preferred). 
OUTPUT: Returns research findings and assigns a confidence_score (0-10)
ROUTES TO: Validator Agent (if confidence < 6) OR Synthesis Agent (if confidence ≥ 6)
c) Validator Agent
Reviews research quality and completeness
Checks if information is sufficient to answer the user's question
OUTPUT: Sets validation_result to "sufficient" or "insufficient"
ROUTES TO:
Research Agent (loop back if insufficient AND attempts < 3)
Synthesis Agent (if sufficient OR max attempts reached)
d) Synthesis Agent
Takes research findings and creates a coherent summary
Formats the response in a user-friendly way
Maintains context from conversation history
ROUTES TO: END

2. Core Features to Implement
Multi-turn Conversation
Maintain conversation history across multiple queries
Each agent should access previous messages for context
Support follow-up questions like "What about their competitors?" or "Tell me more about the CEO"
Human-in-the-Loop (Interrupt)
When Clarity Agent detects an unclear query, interrupt the workflow
Request clarification from the user (e.g., "Which company are you asking about?")
Resume processing after receiving clarification
State Management
Use a proper state schema that includes
Conditional Routing
Appropriate routing across the CLarity, Research & Validation agents

 
