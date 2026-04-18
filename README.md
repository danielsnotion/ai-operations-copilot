# 🤖 AI Operations Copilot — Revenue Anomaly Investigation Agent

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

## 🧠 Core Capabilities

### 🔍 Retrieval-Augmented Generation (RAG)

* FAISS-based vector search
* Dynamic CSV ingestion
* Context-aware responses

---

### 🛠️ Tool-Based Reasoning

* Revenue trend analysis
* Region comparison
* Anomaly detection

---

### 🧠 Memory & Context

* Multi-turn conversation support
* Context retention

---

### 🔁 Feedback Adaptation

* 👍 / 👎 feedback collection
* Improves future responses

---

### 🛡️ Safety Enforcement

* Refuses unsafe requests
* Avoids hallucination
* Explains uncertainty

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

## 🔁 Agent Orchestration

The system supports **two frameworks**:

---

### 🔗 LangChain (Baseline)

```text
Query → Memory → Retrieval → Tool → Response
```

---

### 🔄 LangGraph (Production)

```text
Query
 ↓
Memory
 ↓
Retrieval
 ↓
Planning
 ↓
Tool Selection
 ↓
Tool Execution
 ↓
Response
```

---

## ⚖️ LangChain vs LangGraph

| Feature              | LangChain | LangGraph |
| -------------------- | --------- | --------- |
| Flow                 | Linear    | Stateful  |
| Control              | Limited   | Explicit  |
| Debugging            | Medium    | High      |
| Production readiness | Medium    | High      |

---

## 🏗️ Architecture

```text
Streamlit UI
    ↓
FastAPI Backend
    ↓
Agent (LangChain / LangGraph)
    ↓
Core Components:
    - Memory
    - Retrieval (FAISS)
    - Tools
    - Planner
    - Feedback
```

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

### 🔍 Use Cases

* Revenue anomaly detection
* Trend analysis
* Region comparison
* RAG-based AI applications

---

### 🔗 Direct Download Links

```text
Small:  https://huggingface.co/datasets/daniel1028/ai-operations-copilot-data/blob/main/data_small.csv
Medium: https://huggingface.co/datasets/daniel1028/ai-operations-copilot-data/blob/main/data_medium.csv
Large:  https://huggingface.co/datasets/daniel1028/ai-operations-copilot-data/blob/main/data_large.csv
```

---

## 📂 Project Structure

```text
ai-operations-copilot/
│
├── app/
│   ├── agents/
│   ├── tools/
│   ├── rag/
│   ├── core/
│   └── main.py
│
│
├── cli/
│   └── agent_v2.py
│
├── data/
├── configs/
├── logs/
├── evaluation/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/danielsnotion/ai-operations-copilot
cd ai-operations-copilot

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Run Application

### Start Backend

```bash
uvicorn backend.app.main:app --reload
```

---

### Start UI

```bash
streamlit run ui/app.py
```

---

## 💬 Features

* Chatbot-style UI
* Streaming responses
* Framework selection
* Feedback loop
* Dataset upload & embedding

---

## 🧪 Evaluation

* Prompt comparison (v1 vs v2 vs v3)
* Tool usage validation
* Safety enforcement testing
* Retrieval effectiveness

---

## 🛡️ Safety Design

* No data modification allowed
* Explicit uncertainty handling
* Human escalation support
* No sensitive data logging

---

## ❌ Limitations

* Single-user system
* No authentication service (basic login only)
* Limited statistical modeling

---

## 🚀 Future Improvements

* Advanced anomaly detection
* Distributed vector database
* Multi-user authentication
* Kubernetes deployment

---

## 🏆 Highlights

* Dual-framework agent system
* Full RAG + Tool + Memory pipeline
* Dynamic dataset ingestion
* Production-style architecture

---

## 👨‍💻 Author

**Daniel Arokia**
AI Engineer | Backend Developer

---

## 📄 License

This project is for educational and demonstration purposes.
