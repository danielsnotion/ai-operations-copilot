import os
from dotenv import load_dotenv
from openai import OpenAI
from configs.logging_config import setup_logger
from configs.settings import config
load_dotenv()

logger = setup_logger("llm_agent")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = config["llm"]["model"]


class LLMAgent:
    def __init__(self):
        self.prompts = {
            "v1": self.prompt_v1,
            "v2": self.prompt_v2,
            "v3": self.prompt_v3
        }

    def prompt_v1(self, query):
        return f"""
        You are a business analyst. Answer the question.

        Question: {query}
        """

    def prompt_v2(self, query):
        return f"""
        You are an operations analyst.

        Analyze the user query and provide:
        1. Observations
        2. Possible reasons
        3. Suggested next steps

        Question: {query}
        """

    def prompt_v3(self, query):
        return f"""
        You are an AI Operations Copilot.

        Rules:
        - Do NOT make up data
        - If unsure, say "I don't have enough data"
        - Provide structured reasoning

        Output format:
        - Summary
        - Possible Causes
        - Confidence Level (High/Medium/Low)
        - Suggested Actions

        Question: {query}
        """

    def run(self, query, version="v1"):
        prompt = self.prompts[version](query)

        logger.info(f"Prompt Version: {version}")
        logger.info(f"User Query: {query}")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        output = response.choices[0].message.content

        logger.info(f"Response: {output}")
        return output


def run_cli():
    agent = LLMAgent()

    print("=== LLM Agent ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Enter your query: ")

        if query.lower() == "exit":
            break

        version = input("Choose prompt version (v1/v2/v3): ")

        result = agent.run(query, version)
        print(f"\nAgent Response:\n{result}\n")


if __name__ == "__main__":
    run_cli()