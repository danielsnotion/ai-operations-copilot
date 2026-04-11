import os
from unittest import result
from dotenv import load_dotenv
from openai import OpenAI

from app.agents.crewai_agent import CrewAIAgent
from app.rag.embedding_manager import EmbeddingManager
from app.tools.tool_registry import ToolRegistry
from app.core.memory import ConversationMemory
from app.agents.planner import Planner
from configs.logging_config import setup_logger
from app.core.feedback_store import FeedbackStore
from configs.settings import config
from app.agents.langgraph_agent import langgraph_app
from app.agents.langchain_agent import LangChainAgent
load_dotenv()

logger = setup_logger("agent_v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = config["llm"]["model"]
EMBD_MODEL_NAME = config["embedding"]["model"]
embedding_manager = EmbeddingManager(EMBD_MODEL_NAME)
class AgentV2:
    def __init__(self):
        self.tools = ToolRegistry()
        self.memory = ConversationMemory()
        self.planner = Planner()
        self.feedback_store = FeedbackStore()
        self.embedding_manager = embedding_manager
        self.crewai_agent = CrewAIAgent( self.tools, self.memory, self.planner, self.embedding_manager, self.feedback_store )
        self.langchain_agent = LangChainAgent( self.tools, self.memory, self.planner, self.embedding_manager, self.feedback_store )

    def run(self, query, llm_model, framework="LangGraph", api_key=None, auth_mode=None):
        logger.info(f"[Framework Selected]: {framework}")
        if auth_mode == "external" and api_key:
            logger.info("Using external API key for OpenAI client")
            os.environ["OPENAI_API_KEY"] = api_key

        if framework == "LangGraph":
            logger.info("Running with LangGraph Agent")
            return self.run_langgraph(query, llm_model)
        
        if framework == "CrewAI":
            logger.info("Running with CrewAI Agent")
            return self.crewai_agent.run(query, llm_model)

        elif framework == "LangChain":
            logger.info("Running with LangChain Agent")
            return self.run_langchain(query, llm_model)
        else:
            logger.warning(f"Unknown framework: {framework}, defaulting to LangGraph")
            return self.run_langgraph(query, llm_model)
    
   
    
    def run_langgraph(self, query, llm_model=MODEL_NAME):

        result = langgraph_app.invoke({
            "query": query,
            "llm_model": llm_model
        },
        config={"recursion_limit": 10})

        return {
            "final_answer": result.get("final_answer"),
            "trace": result.get("trace", [])
        }

    def run_langchain(self, query, llm_model=MODEL_NAME):
        return self.langchain_agent.run(query, llm_model)
        

def run_cli():
    agent = AgentV2()

    print("=== Agent V2 (Memory + Planning) ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Enter your query: ")

        if query.lower() == "exit":
            break

        result = agent.run(query)
        print(f"\nAgent Response:\n{result}\n")

        feedback = input("Was this helpful? (yes/no): ")

        if feedback.lower() == "yes":
            agent.feedback_store.save_feedback(query, result, "positive")
        else:
            agent.feedback_store.save_feedback(query, result, "negative")


if __name__ == "__main__":
    run_cli()