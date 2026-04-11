import os
from urllib import response
from dotenv import load_dotenv
from openai import OpenAI
from app.tools.tool_registry import ToolRegistry
from configs.logging_config import setup_logger
from configs.settings import config

load_dotenv()

logger = setup_logger("tool_agent")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = config["llm"]["model"]


class ToolAgent:
    def __init__(self):
        self.registry = ToolRegistry()
    def decide_tool_llm(self, query, llm_model=MODEL_NAME):
        tools = [
            "analyze_revenue_trend",
            "compare_regions",
            "detect_anomalies"
        ]

        prompt = f"""
    You are an AI routing agent.

    Available tools:
    - analyze_revenue_trend: Detect revenue increase or decrease
    - compare_regions: Compare performance across regions
    - detect_anomalies: Find unusual revenue patterns

    User Query:
    {query}

    Return ONLY one tool name from the list above.
    If no tool is relevant, return "NONE".
    """

        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )

        tool_name = response.choices[0].message.content.strip()

        # ✅ Normalize output
        tool_name = tool_name.replace('"', '').strip()

        # ✅ Validate tool
        if tool_name not in tools:
            logger.warning(f"Invalid tool from LLM: {tool_name}")
            return None

        return tool_name

    def decide_tool(self, query):
        query = query.lower()

        keywords = {
    "analyze_revenue_trend": ["trend", "drop", "increase"],
    "compare_regions": ["region", "area"],
    "detect_anomalies": ["anomaly", "anomalies", "outlier", "unusual"]
}

        for tool, words in keywords.items():
            if any(word in query for word in words):
                return tool

    def run(self, query):
        logger.info(f"User Query: {query}")

        tool_name = self.decide_tool_llm(query)

        logger.info(f"Selected Tool: {tool_name}")

        if tool_name is None:
            logger.warning("No tool matched. Falling back to LLM.")

        if tool_name:
            tool = self.registry.get_tool(tool_name)
            tool_result = tool()

            logger.info(f"Tool Result: {tool_result}")

            prompt = f"""
You are an AI Operations Copilot.

User Question:
{query}

Tool Output:
{tool_result}

Explain clearly using the tool output.
Provide:
- Summary
- Key Insight
- Suggested Action
"""

        else:
            prompt = f"""
User Question:
{query}

Respond normally.
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content
        logger.info(f"Response: {output}")

        return output


def run_cli():
    agent = ToolAgent()

    print("=== Tool Agent ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Enter your query: ")

        if query.lower() == "exit":
            break

        result = agent.run(query)
        print(f"\nAgent Response:\n{result}\n")


if __name__ == "__main__":
    run_cli()