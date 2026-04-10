import pandas as pd
from configs.logging_config import setup_logger

# Initialize logger
logger = setup_logger("baseline_agent")


class BaselineAgent:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)

    def handle_query(self, query: str) -> str:
        query_lower = query.lower()
        logger.info(f"User Query: {query}")

        # Rule-based responses (intentionally simplistic)

        if "revenue drop" in query_lower:
            response = "Revenue dropped recently. Please check regional performance."
        
        elif "region" in query_lower:
            response = "North and South regions are contributing to revenue changes."

        elif "refund" in query_lower:
            response = "Refunds seem higher than usual."

        elif "why" in query_lower:
            response = "The system cannot determine the exact cause in this version."

        else:
            response = "Sorry, I cannot understand your query. Please rephrase."

        logger.info(f"Agent Response: {response}")
        return response


def run_cli():
    agent = BaselineAgent("data/sales_data.csv")

    print("=== Revenue Copilot (Baseline Version) ===")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Enter your query: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = agent.handle_query(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    run_cli()