from langgraph.graph import StateGraph
from typing import TypedDict, List, Any

from app.rag.embedding_manager import EmbeddingManager
from app.tools.tool_registry import ToolRegistry
from app.core.memory import ConversationMemory
from app.agents.planner import Planner
from app.core.feedback_store import FeedbackStore
from configs.settings import config
from configs.logging_config import setup_logger
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger("langgraph_agent")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding_manager = EmbeddingManager()

# ---------------- STATE ---------------- #
class AgentState(TypedDict):
    query: str
    context: str
    retrieved_data: str
    plan: str
    tool: str
    tool_result: str
    final_answer: str
    llm_model: str
    trace: List[str]


# ---------------- COMPONENTS ---------------- #
tools = ToolRegistry()
memory = ConversationMemory()
planner = Planner()
feedback_store = FeedbackStore()


# ---------------- HELPERS ---------------- #
def add_trace(state: AgentState, message: str):
    state["trace"].append(message)
    logger.info(f"[TRACE] {message}")


# ---------------- NODES ---------------- #

def load_memory(state: AgentState):
    state["trace"] = []
    add_trace(state, "Started execution")

    state["context"] = memory.get_context()
    add_trace(state, "Loaded conversation memory")

    return state


def retrieve(state: AgentState):
    docs = embedding_manager.search(state["query"])
    if not embedding_manager.is_ready():
            state["trace"].append("Vector DB not initialized")
            docs = []
    else:
        docs = embedding_manager.search(state["query"])

    if not docs:
        state["trace"].append("No relevant documents found (empty vector DB)")

    context = "\n".join(docs) if docs else "No additional context available."
    state["retrieved_data"] = context
    
    add_trace(state, f"Retrieved {len(docs)} records from vector DB")

    return state


def plan(state: AgentState):
    state["plan"] = planner.create_plan(
        state["query"],
        state["context"],
        state["llm_model"]
    )

    add_trace(state, f"Planned action: {state['plan']}")

    return state


def decide_tool(state: AgentState):

    prompt = f"""
Available tools:
- analyze_revenue_trend
- compare_regions
- detect_anomalies

User Query:
{state["query"]}

Return ONLY tool name or NONE.
"""

    response = client.chat.completions.create(
        model=state["llm_model"],
        messages=[{"role": "user", "content": prompt}]
    )

    tool_name = response.choices[0].message.content.strip()

    if tool_name not in [
        "analyze_revenue_trend",
        "compare_regions",
        "detect_anomalies"
    ]:
        tool_name = None

    state["tool"] = tool_name

    add_trace(state, f"Tool selected: {tool_name}")

    return state


def execute_tool(state: AgentState):

    if state["tool"]:
        tool_fn = tools.get_tool(state["tool"])
        result = tool_fn()

        state["tool_result"] = result
        add_trace(state, f"Tool executed: {state['tool']}")

    else:
        state["tool_result"] = None
        add_trace(state, "No tool executed")

    return state


def generate_response(state: AgentState):

    negative_cases = feedback_store.get_negative_cases()

    feedback_context = ""
    if negative_cases:
        feedback_context = "Previous mistakes to avoid:\n"
        for case in negative_cases[-3:]:
            feedback_context += f"- {case['query']}\n"

    prompt = f"""
You are an AI Operations Copilot.

Conversation:
{state["context"]}

Retrieved Data:
{state["retrieved_data"]}

Plan:
{state["plan"]}

Previous Mistakes:
{feedback_context}

User Query:
{state["query"]}

Tool Output:
{state["tool_result"]}

Provide:
- Final Answer
- Explanation
- Confidence Level

If no context is available, answer based on general reasoning.

"""

    response = client.chat.completions.create(
        model=state["llm_model"],
        messages=[{"role": "user", "content": prompt}]
    )

    output = response.choices[0].message.content

    state["final_answer"] = output

    memory.add(state["query"], output)

    add_trace(state, "Generated final response using LLM")

    return state


# ---------------- GRAPH ---------------- #

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_memory", load_memory)
    graph.add_node("retrieve", retrieve)
    graph.add_node("plan", plan)
    graph.add_node("decide_tool", decide_tool)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("respond", generate_response)

    graph.set_entry_point("load_memory")

    graph.add_edge("load_memory", "retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "decide_tool")
    graph.add_edge("decide_tool", "execute_tool")
    graph.add_edge("execute_tool", "respond")

    return graph.compile()


langgraph_app = build_graph()