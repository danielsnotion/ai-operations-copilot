from openai import OpenAI
from configs.settings import config
from configs.logging_config import setup_logger

logger = setup_logger("langchain_agent")

client = OpenAI()

MODEL_NAME = config["llm"]["model"]


class LangChainAgent:

    def __init__(self, tools, memory, planner, embedding_manager, feedback_store):
        self.tools = tools
        self.memory = memory
        self.planner = planner
        self.embedding_manager = embedding_manager
        self.feedback_store = feedback_store

    # ---------------- TRACE HELPER ---------------- #
    def add_trace(self, trace, message):
        trace.append(message)
        logger.info(f"[TRACE] {message}")

    # ---------------- MAIN RUN ---------------- #
    def run(self, query, llm_model=MODEL_NAME):

        trace = []

        logger.info(f"[LangChain] Query: {query}")

        # -------- STEP 1: MEMORY -------- #
        context = self.memory.get_context()
        self.add_trace(trace, "Loaded conversation memory")

        # -------- STEP 2: RETRIEVAL -------- #
        docs = self.embedding_manager.search(query)
        rag_context = "\n".join(docs)
        self.add_trace(trace, f"Retrieved {len(docs)} records from vector DB")

        # -------- STEP 3: PLANNING -------- #
        plan = self.planner.create_plan(query, context, llm_model)
        self.add_trace(trace, f"Planned action: {plan}")

        # -------- STEP 4: TOOL SELECTION -------- #
        tool_name = self.decide_tool(query, llm_model)
        self.add_trace(trace, f"Tool selected: {tool_name}")

        # -------- STEP 5: TOOL EXECUTION -------- #
        tool_result = None
        if tool_name:
            try:
                tool = self.tools.get_tool(tool_name)
                tool_result = tool()
                self.add_trace(trace, f"Tool executed: {tool_name}")
            except Exception as e:
                tool_result = f"Tool error: {str(e)}"
                self.add_trace(trace, f"Tool execution failed: {tool_name}")
        else:
            self.add_trace(trace, "No tool executed")

        # -------- STEP 6: FEEDBACK CONTEXT -------- #
        negative_cases = self.feedback_store.get_negative_cases()

        feedback_context = ""
        if negative_cases:
            feedback_context = "Previous mistakes to avoid:\n"
            for case in negative_cases[-3:]:
                feedback_context += f"- {case['query']}\n"

            self.add_trace(trace, f"Loaded {len(negative_cases)} feedback signals")

        # -------- STEP 7: FINAL RESPONSE -------- #
        prompt = f"""
You are an AI Operations Copilot.

Conversation:
{context}

Retrieved Data:
{rag_context}

Plan:
{plan}

Previous Mistakes:
{feedback_context}

User Query:
{query}

Tool Output:
{tool_result}

Instructions:
- Use retrieved data when relevant
- Do NOT hallucinate
- Combine tool output + retrieved data

Provide:
- Final Answer
- Explanation
- Confidence Level
"""

        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content

        self.add_trace(trace, "Generated final response using LLM")

        # -------- STEP 8: MEMORY UPDATE -------- #
        self.memory.add(query, output)
        self.add_trace(trace, "Updated conversation memory")

        logger.info(f"[LangChain] Response: {output[:200]}")

        return {
            "final_answer": output,
            "trace": trace
        }

    # ---------------- TOOL DECISION ---------------- #
    def decide_tool(self, query, llm_model):

        prompt = f"""
Available tools:
- analyze_revenue_trend
- compare_regions
- detect_anomalies

User Query:
{query}

Return ONLY tool name or NONE.
"""

        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )

        tool_name = response.choices[0].message.content.strip()

        valid_tools = [
            "analyze_revenue_trend",
            "compare_regions",
            "detect_anomalies"
        ]

        if tool_name not in valid_tools:
            return None

        return tool_name