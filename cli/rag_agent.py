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
        if not docs:
            return "No additional context available"

        # Handle system messages
        if len(docs) == 1 and docs[0] in [
            "Vector DB not initialized",
            "No relevant documents found",
            "Empty query"
        ]:
            return "No additional context available"

        return "\n\n".join([f"- {doc}" for doc in docs])

    def generate_prompt(self, query, context, use_rag=True):
        if not use_rag:
            return f"""
You are an AI Operations Copilot.

Answer based on general knowledge.

Question:
{query}

Output format:
- Summary
- Key Insights
- Confidence Level
- Suggested Actions
"""

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

    def run(self, query, use_rag=True):
        logger.info(f"User Query: {query}")

        # 🔹 WITHOUT RAG
        if not use_rag:
            prompt = self.generate_prompt(query, context=None, use_rag=False)

        # 🔹 WITH RAG
        else:
            retrieved_docs = self.retriever.retrieve(query)
            logger.info(f"Retrieved Docs: {retrieved_docs}")

            context = self.format_context(retrieved_docs)
            logger.info(f"Context: {context}")

            prompt = self.generate_prompt(query, context, use_rag=True)

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

        print("\n--- WITHOUT RAG ---")
        print(agent.run(query, use_rag=False))

        print("\n--- WITH RAG ---")
        print(agent.run(query, use_rag=True))
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    run_cli()