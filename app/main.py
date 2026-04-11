from fastapi import FastAPI
from pydantic import BaseModel
import time

from configs.logging_config import setup_logger
from configs.settings import config
from fastapi import UploadFile, File

app = FastAPI()
logger = setup_logger("fastapi")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = None
embedding_manager = None

def get_agent():
    global agent
    if agent is None:
        from cli.agent_v2 import AgentV2
        agent = AgentV2()
    return agent

def get_embedding_manager():
    global embedding_manager
    if embedding_manager is None:
        from app.rag.embedding_manager import EmbeddingManager
        embedding_manager = EmbeddingManager()
    return embedding_manager

MODEL_NAME = config["llm"]["model"]
EMBD_MODEL_NAME = config["embedding"]["model"]


class QueryRequest(BaseModel):
    query: str
    llm_model: str = MODEL_NAME
    framework: str = "LangGraph"
    api_key: str | None = None
    auth_mode: str | None = None


@app.post("/ask")
def ask_agent(request: QueryRequest):
    start_time = time.time()
    agent = get_agent()
    try:
        response = agent.run(request.query,
                             llm_model=request.llm_model,
                             framework=request.framework,
                             api_key=request.api_key,
                             auth_mode=request.auth_mode
                             )
        latency = time.time() - start_time

        logger.info(f"Latency: {latency:.2f}s")

        return {
            "response": response,
            "latency": latency
        }

    except Exception as e:
        logger.exception("Unhandled error occurred")

        return {
            "response": "The system encountered an issue. Please try again later.",
            "error": "internal_error"
        }

from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    query: str
    response: str
    feedback: str


@app.post("/feedback")
def save_feedback(request: FeedbackRequest):
    agent = get_agent()
    agent.feedback_store.save_feedback(
        request.query,
        request.response,
        request.feedback
    )
    return {"status": "saved"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())
    embedding_manager = get_embedding_manager()
    embedding_manager.add_csv(file_path, file.filename)

    return {"message": "Embedding created successfully"}

@app.get("/metadata")
def get_metadata():
    import json
    try:
        return json.load(open("data/metadata.json"))
    except:
        return []