# 🤖 AI Operations Copilot — Revenue Anomaly Investigation Agent

An **Agentic AI system** for business insights using **LangGraph, LangChain, and CrewAI**, powered by **RAG (Retrieval-Augmented Generation)** and enhanced with **LangSmith observability**.

---

## 📌 Overview

This project implements a **production-grade AI Operations Copilot** that assists business analysts in investigating revenue anomalies.

It combines:

* LLM reasoning
* Retrieval-Augmented Generation (RAG)
* Tool-based analytics
* Conversation memory
* Feedback-driven adaptation
* Orchestration (LangChain, LangGraph, CrewAI)

---

## 🎯 Problem Statement

Business teams struggle to quickly diagnose:

* Revenue drops
* Regional performance issues
* Refund anomalies

Manual workflows are slow, inconsistent, and lack explainability.

---

## ✅ Solution

An AI Copilot that:

* Provides **data-driven insights**
* Uses **tools for accurate computation**
* Maintains **context-aware conversations**
* Adapts using **user feedback**
* Enforces **safety-first behavior**

---

## 👤 User Persona

**Primary User:** Business / Operations Analyst

### Workflow:

* Ask analytical questions
* Investigate anomalies
* Validate insights
* Generate reports

---

## 🚀 Core Capabilities & Features

### 🧠 Multi-Agent Framework Support

* **LangGraph** → State machine orchestration
* **LangChain** → Structured reasoning pipeline
* **CrewAI** → Multi-agent collaboration

### 📊 Data-Driven Intelligence (RAG)

* Upload CSV datasets
* Automatic embedding generation
* Semantic search using vector database
* Context-aware answers

### 🛠️ Tool-Based Reasoning

* Revenue trend analysis
* Region comparison
* Anomaly detection

### 🧠 Memory & Context

* Multi-turn conversation support
* Context retention

### 🔁 Feedback Adaptation

* 👍 / 👎 feedback collection
* Improves future responses

### 🛡️ Safety Enforcement

* Refuses unsafe requests
* Avoids hallucination
* Explains uncertainty

### 💬 ChatGPT-like UI

* Streaming responses (real-time tokens)
* Thinking state (🤖 Thinking...)
* Auto-scroll + chat layout
* Feedback system (👍 / 👎)
* Trace (reasoning panel)

### 🔍 Observability (LangSmith)

* Full execution trace
* Tool usage tracking
* Latency monitoring
* Debuggable workflows

### 🔐 Flexible Access

* Login mode (internal API key)
* External API key support

### 🔍 Use Cases

* Revenue anomaly detection
* Trend analysis
* Region comparison
* RAG-based AI applications

---

## 🏗️ Architecture

```
UI (Streamlit)
   ↓
FastAPI Backend
   ↓
AgentV2
   ├── LangGraph Agent
   ├── LangChain Agent
   └── CrewAI Agent
   ↓
Core Components:
    - Memory
    - Embedding Manager (FAISS)
    - Tools
    - Planner
    - Feedback
   ↓
CSV Data (RAG)
```

---

## ⚙️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: FastAPI
* **LLM**: OpenAI
* **Frameworks**: LangChain, LangGraph, CrewAI
* **Vector DB**: FAISS
* **Observability**: LangSmith
* **Storage**: Local CSV + Metadata

---

---

## 📊 Dataset

Synthetic datasets are provided via Hugging Face:

👉 **Dataset Repository:**
https://huggingface.co/datasets/daniel1028/ai-operations-copilot-data

---

### 📁 Available Datasets

| Dataset | Size     | Purpose             |
| ------- | -------- | ------------------- |
| Small   | 100 rows | Quick demo          |
| Medium  | 10K rows | Functional testing  |
| Large   | 1M rows  | Performance testing |

---

### 📊 Schema

| Column           | Description      |
| ---------------- | ---------------- |
| date             | Transaction date |
| region           | Sales region     |
| product          | Product name     |
| category         | Product category |
| orders           | Number of orders |
| revenue          | Total revenue    |
| refunds          | Refund count     |
| price            | Unit price       |
| customer_segment | Customer type    |


---

### 🔗 Direct Download Links

```text
Small:  https://huggingface.co/datasets/daniel1028/ai-operations-copilot-data/blob/main/data_small.csv
Medium: https://huggingface.co/datasets/daniel1028/ai-operations-copilot-data/blob/main/data_medium.csv
Large:  https://huggingface.co/datasets/daniel1028/ai-operations-copilot-data/blob/main/data_large.csv
```

---

## 📂 Project Structure (Final)

```
.
├── app/
│   ├── agents/
│   │   ├── langchain_agent.py
│   │   ├── langgraph_agent.py
│   │   └── crewai_agent.py
│   ├── rag/
│   │   └── embedding_manager.py
│   ├── core/
│   │   ├── memory.py
│   │   └── feedback_store.py
│   ├── tools/
│   │   └── tool_registry.py
│   └── main.py
├── cli/
│   └── agent_v2.py
├── evaluation/
│   ├── run_tests.py        # Test cases with sample queries
│   └── test_cases.json     # Sample Inputs
│   ├── results/
│       └── results.json.   # results generated by test cases
├── data/
│   └── metadata.json
├── requirements.txt
└── README.md
```

---

## 🧪 Setup Instructions

### 1️⃣ Clone Repository

```
git clone https://github.com/danielsnotion/ai-operations-copilot.git
cd ai-operations-copilot
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment Variables

Create `.env`:

```
OPENAI_API_KEY=your_openai_key

# LangSmith (IMPORTANT)
LANGCHAIN_API_KEY=lsv2_xxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-operations-copilot
```

---

## ▶️ Run Application

### 🔹 Backend (FastAPI)

```
uvicorn app.main:app --reload
```

---

## 📊 Data Workflow

1. Upload CSV in **Data Tab**
2. Embeddings generated automatically
3. Vector DB updated
4. Query via Chat tab

---

## 🔄 API Endpoints

| Endpoint      | Description        |
| ------------- | ------------------ |
| `/ask`        | Standard response  |
| `/ask-stream` | Streaming response |
| `/upload`     | Upload dataset     |
| `/metadata`   | View datasets      |
| `/feedback`   | Store feedback     |

---

## 🧠 LangSmith Observability

View traces at:

👉 https://smith.langchain.com/o/e817ca77-f465-43aa-be36-6391f11645fe/projects/p/32f2d7c6-10cc-4434-914c-e92abb3289b4

Includes:

* Execution path
* Tool usage
* LLM calls
* Latency

---

## 🔐 Authentication & API Key Handling

The system supports **two access modes**:

### 1️⃣ Login Mode (Demo Mode)

* User logs in with:

  ```
  Username: daniel
  Password: <CONTACT ME>
  ```
* System uses **internal OpenAI API key**

---

### 2️⃣ API Key Mode (User Mode)

* User provides their own OpenAI API key
* System uses user-provided key for inference

---

### 🔒 Security Notes

* API keys are **never logged**
* External keys are stored **only in session**
* No sensitive data is persisted

---

## ⚠️ Troubleshooting

### ❌ LangSmith 401 Error

* Ensure `LANGCHAIN_API_KEY` starts with `lsv2_`
* Verify `.env` is loaded

---

### ❌ No Data / Empty Response

* Upload dataset in Data tab
* Ensure embeddings generated

---

### ❌ Slow Performance

* First run loads embedding model
* Subsequent runs are faster

---

## 🎯 Future Enhancements

* 🔄 Real-time trace streaming in UI
* 📊 Multi-dataset switching
* 💾 Persistent FAISS index
* 📈 Evaluation pipeline (LangSmith)
* 🤖 Auto-agent selection

---

## 👨‍💻 Author

**Daniel Arokia**

* GitHub: https://github.com/danielsnotion

---

## ⭐ If you found this useful

Give a ⭐ on GitHub — it helps!
