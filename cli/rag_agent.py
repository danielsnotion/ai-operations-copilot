import os
from dotenv import load_dotenv
from openai import OpenAI
from app.rag.retriever import DataRetriever
from configs.logging_config import setup_logger
from configs.settings import config

load_dotenv()

logger = setup_logger("rag_agent")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = config["llm"]["model"]

class RAGAgent:
    def __init__(self):
        self.retriever = DataRetriever("data/sales_data.csv")

    def format_context(self, docs):
        return "\n\n".join([f"- {doc}" for doc in docs])

    def generate_prompt(self, query, context):
        return f"""
You are an AI Operations Copilot.

Use ONLY the provided data to answer.
If insufficient data, say so.

Context:
{context}

Question:
{query}

Output format:
- Summary
- Key Insights
- Confidence Level
- Suggested Actions
"""

    def run(self, query):
        logger.info(f"User Query: {query}")

        retrieved_docs = self.retriever.retrieve(query)
        logger.info(f"Retrieved Docs: {retrieved_docs}")

        context = self.format_context(retrieved_docs)

        logger.info(f"Retrieved Context: {context}")

        prompt = self.generate_prompt(query, context)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        output = response.choices[0].message.content
        logger.info(f"Response: {output}")

        return output


def run_cli():
    agent = RAGAgent()

    print("=== RAG Agent ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Enter your query: ")

        if query.lower() == "exit":
            break

        result = agent.run(query)
        print(f"\nAgent Response:\n{result}\n")


if __name__ == "__main__":
    run_cli()