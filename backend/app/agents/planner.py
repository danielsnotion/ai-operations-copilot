from openai import OpenAI
from dotenv import load_dotenv
import os
from configs.settings import config
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = config["llm"]["model"]


class Planner:
    def create_plan(self, query, context, llm_model=MODEL_NAME):
        prompt = f"""
You are an AI planning agent.

Conversation Context:
{context}

User Query:
{query}

Break this into steps:
1. Understand intent
2. Decide tool (if needed)
3. Explain reasoning

Return in structured format:
- Intent
- Tool
- Reason
"""

        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content