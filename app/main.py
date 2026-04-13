import os

from fastapi import FastAPI
from pydantic import BaseModel
import time

from configs.logging_config import setup_logger
from configs.settings import config
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
logger = setup_logger("fastapi")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-operations-copilot"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

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
        emb_mgr = get_embedding_manager()
        agent = AgentV2(embedding_manager=emb_mgr)
    return agent

def get_embedding_manager():
    global embedding_manager
    if embedding_manager is None:
        from app.rag.embedding_manager import EmbeddingManager
        embedding_manager = EmbeddingManager()
    return embedding_manager

def reset_services():
    global agent, embedding_manager
    agent = None
    embedding_manager = None

MODEL_NAME = config["llm"]["model"]
EMBD_MODEL_NAME = config["embedding"]["model"]


class QueryRequest(BaseModel):
    query: str
    llm_model: str = MODEL_NAME
    framework: str = "LangGraph"
    api_key: str | None = None
    auth_mode: str | None = None

class FeedbackRequest(BaseModel):
    query: str
    response: str
    feedback: str

@app.post("/ask")
def ask_agent(request: QueryRequest):
    start_time = time.time()
    agent = get_agent()
    try:
        if not request.query.strip():
            latency = time.time() - start_time
            logger.info(f"Latency: {latency:.2f}s")
            return {"response": "Query cannot be empty",
                    "latency": latency}

        if len(request.query) > 1000:
            latency = time.time() - start_time
            logger.info(f"Latency: {latency:.2f}s")
            return {"response": "Query too long",
                    "latency": latency}

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



@app.post("/ask-stream")
def ask_agent_stream(request: QueryRequest):

    def generate():
        agent = get_agent()

        for chunk in agent.run_stream(
            request.query,
            llm_model=request.llm_model,
            framework=request.framework,
            api_key=request.api_key,
            auth_mode=request.auth_mode
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

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

     # Reset everything BEFORE rebuilding
    reset_services()
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