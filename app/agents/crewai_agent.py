from crewai import Agent, Task, Crew
from openai import OpenAI
from configs.settings import config
from configs.logging_config import setup_logger

logger = setup_logger("crewai_agent")
client = OpenAI()

MODEL_NAME = config["llm"]["model"]


class CrewAIAgent:

    def __init__(self, tools, memory, planner, embedding_manager, feedback_store):
        self.tools = tools
        self.memory = memory
        self.planner = planner
        self.embedding_manager = embedding_manager
        self.feedback_store = feedback_store

    # ---------------- TRACE ---------------- #
    def add_trace(self, trace, message):
        trace.append(message)
        logger.info(f"[TRACE] {message}")

    # ---------------- RUN ---------------- #
    def run(self, query, llm_model=MODEL_NAME):

        trace = []

        logger.info(f"[CrewAI] Query: {query}")

        # -------- MEMORY -------- #
        context = self.memory.get_context()
        self.add_trace(trace, "Loaded conversation memory")

        # -------- RETRIEVAL -------- #
        docs = self.embedding_manager.search(query)
        rag_context = "\n".join(docs)
        self.add_trace(trace, f"Retrieved {len(docs)} records from vector DB")

        # -------- FEEDBACK -------- #
        negative_cases = self.feedback_store.get_negative_cases()
        feedback_context = ""
        if negative_cases:
            feedback_context = "\n".join([f"- {c['query']}" for c in negative_cases[-3:]])
            self.add_trace(trace, f"Loaded {len(negative_cases)} feedback signals")

        # ================= AGENTS ================= #

        planner_agent = Agent(
            role="Planner",
            goal="Understand user intent and create a plan",
            backstory="Expert in breaking down analytical problems",
            allow_delegation=False,
            verbose=True
        )

        tool_agent = Agent(
            role="Tool Executor",
            goal="Select and execute the right tool",
            backstory="Expert in data tools and execution",
            allow_delegation=False,
            verbose=True
        )

        analyst_agent = Agent(
            role="Analyst",
            goal="Generate final answer using all context",
            backstory="Expert business analyst",
            allow_delegation=False,
            verbose=True
        )

        # ================= TASKS ================= #

        plan_task = Task(
            description=f"""
                User Query: {query}

                Conversation Context:
                {context}

                Retrieved Data:
                {rag_context}

                Create a clear step-by-step plan to answer the query.
                """,
                    expected_output="A structured plan explaining how to solve the user query",
                    agent=planner_agent
                    )

        tool_task = Task(
            description=f"""
                Available tools:
                - analyze_revenue_trend
                - compare_regions
                - detect_anomalies

                User Query:
                {query}

                Decide which tool to use and execute it.
                Return tool name and result.
                """,
                    expected_output="Selected tool name and execution result",
                    agent=tool_agent
        )

        final_task = Task(
            description=f"""
                User Query: {query}

                Conversation:
                {context}

                Retrieved Data:
                {rag_context}

                Previous Mistakes:
                {feedback_context}

                Combine all information and provide:

                - Final Answer
                - Explanation
                - Confidence Level
                """,
                    expected_output="Final structured answer with explanation and confidence level",
                    agent=analyst_agent
        )

        # ================= CREW ================= #
        crew = Crew(
            agents=[planner_agent, tool_agent, analyst_agent],
            tasks=[plan_task, tool_task, final_task],
            verbose=True
        )

        # -------- EXECUTE -------- #
        result = crew.kickoff()

        self.add_trace(trace, "Planner agent executed")
        self.add_trace(trace, "Tool agent executed")
        self.add_trace(trace, "Analyst agent generated response")

        # -------- MEMORY UPDATE -------- #
        self.memory.add(query, str(result))
        self.add_trace(trace, "Updated conversation memory")

        return {
            "final_answer": str(result),
            "trace": trace
        }