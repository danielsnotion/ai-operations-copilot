import streamlit as st
import uuid
import time
import os
import json
from dotenv import load_dotenv

from cli.agent_v2 import AgentV2
from backend.app.rag.embedding_manager import EmbeddingManager
load_dotenv()
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(layout="wide")
st.title("🤖 AI Operations Copilot")

# ---------------- INIT ---------------- #


@st.cache_resource
def get_embedding_manager(version=0):
    return EmbeddingManager()

@st.cache_resource
def get_agent():
    return AgentV2()


#----------------- Session State ---------------- #
if "embedding_version" not in st.session_state:
    st.session_state.embedding_version = 0

agent = get_agent()
embedding_manager = get_embedding_manager(st.session_state.embedding_version)


# ---------------- TABS ---------------- #
tab1, tab2 = st.tabs(["💬 Chat", "📂 Data"])

# ---------------- AUTO SCROLL ---------------- #
def auto_scroll():
    st.markdown("""
    <script>
        const parent = window.parent.document;
        const scrollContainer = parent.querySelector('section.main');
        if (scrollContainer) {
            scrollContainer.scrollTo({
                top: scrollContainer.scrollHeight,
                behavior: 'smooth'
            });
        }
    </script>
    """, unsafe_allow_html=True)

# ---------------- SESSION ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None

if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "pending_response" not in st.session_state:
    st.session_state.pending_response = None

if "feedback_status" not in st.session_state:
    st.session_state.feedback_status = {}

if "streaming" not in st.session_state:
    st.session_state.streaming = False

# ---------------- CSS FIX ---------------- #
st.markdown("""
<style>
.block-container {
    padding-bottom: 80px !important;
}
section.main {
    scroll-behavior: smooth;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Configuration")

llm_model = st.sidebar.selectbox(
    "LLM Model",
    ["gpt-4.1-mini", "gpt-4o-mini"]
)

framework = st.sidebar.selectbox(
    "Agent Framework",
    ["LangChain","LangGraph", "CrewAI"]
)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.pending_response = None

# ---------------- AUTH ---------------- #
st.sidebar.header("🔐 Access")

mode = st.sidebar.radio("Choose Mode", ["Login", "Use API Key"])

if mode == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "daniel" and password == os.getenv("APP_LOGIN_KEY"):
            st.session_state.auth_mode = "internal"
            st.success("Logged in")
        else:
            st.error("Invalid credentials")

elif mode == "Use API Key":
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    if st.sidebar.button("Use Key"):
        if api_key:
            st.session_state.auth_mode = "external"
            st.session_state.api_key = api_key
            st.success("Using provided API key")

# ---------------- STREAMING ---------------- #
def stream_text(text, delay=0.005):
    placeholder = st.empty()
    streamed = ""

    for char in text:
        streamed += char
        placeholder.markdown(streamed + "▌")
        time.sleep(delay)
        auto_scroll()

    placeholder.markdown(streamed)
    auto_scroll()

    return streamed

# ================= CHAT TAB ================= #
with tab1:

    # APPLY PENDING RESPONSE
    if st.session_state.pending_response:
        st.session_state.messages.append(st.session_state.pending_response)
        st.session_state.pending_response = None

    # CHAT HISTORY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("latency") is not None:
                st.caption(f"⚡ Response time: {msg['latency']:.2f} sec")

            # st.caption(f"Embedding Version: {st.session_state.embedding_version}")

            if msg.get("trace"):
                with st.expander("🧠 Reasoning (Trace)"):
                    for step in msg["trace"]:
                        st.markdown(f"• {step}")

            # -------- FEEDBACK -------- #
            if msg["role"] == "assistant":
                feedback = msg.get("feedback")
                temp_feedback = st.session_state.feedback_status.get(msg["id"])

                if feedback is None and temp_feedback is None:
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("👍 Helpful", key=f"up_{msg['id']}"):
                            agent.feedback_store.save_feedback(
                                msg["query"], msg["content"], "positive"
                            )
                            msg["feedback"] = "positive"
                            st.session_state.feedback_status[msg["id"]] = "positive"
                            st.rerun()

                    with col2:
                        if st.button("👎 Not Helpful", key=f"down_{msg['id']}"):
                            agent.feedback_store.save_feedback(
                                msg["query"], msg["content"], "negative"
                            )
                            msg["feedback"] = "negative"
                            st.session_state.feedback_status[msg["id"]] = "negative"
                            st.rerun()

                elif temp_feedback:
                    st.success("Feedback saved!")
                    del st.session_state.feedback_status[msg["id"]]

                elif feedback:
                    st.caption(f"Feedback: {feedback}")

    # STREAMING
    if st.session_state.streaming:
        with st.chat_message("assistant"):
            st.caption(f"⚙️ {st.session_state.framework} | {st.session_state.llm_model}")
            streamed_text = stream_text(st.session_state.streaming_text)

        st.session_state.pending_response = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": streamed_text,
            "query": st.session_state.last_query,
            "feedback": None,
            "trace": st.session_state.last_trace,
            "latency": st.session_state.last_latency
        }

        st.session_state.streaming = False
        st.rerun()

    # INPUT
    query = st.chat_input("Ask your question...", disabled=st.session_state.streaming)

    if query:

        if not st.session_state.auth_mode:
            st.warning("⚠️ Please login or provide API key first")
            st.stop()

        st.session_state.messages.append({
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": query
        })

        auto_scroll()

        # ✅ DIRECT AGENT CALL (NO FASTAPI)
        start = time.time()

        result = agent.run(
            query=query,
            llm_model=llm_model,
            framework=framework,
            api_key=st.session_state.api_key,
            auth_mode=st.session_state.auth_mode
        )

        latency = time.time() - start

        if "response" in result:
            # FastAPI format
            api_payload = result.get("response", {})

            if isinstance(api_payload, dict):
                answer = api_payload.get("final_answer", "")
                trace = api_payload.get("trace", [])
            else:
                answer = api_payload
                trace = []

            latency = result.get("latency")

        else:
            # Direct Agent call (HF mode)
            answer = result.get("final_answer", "")
            trace = result.get("trace", [])
            latency = None

        st.write("DEBUG:", result)

        st.session_state.streaming = True
        st.session_state.streaming_text = answer
        st.session_state.last_query = query
        st.session_state.last_trace = trace
        st.session_state.last_latency = latency
        st.session_state.framework = framework
        st.session_state.llm_model = llm_model

        st.rerun()

# ================= DATA TAB ================= #
with tab2:

    st.header("📂 Upload CSV & Generate Embeddings")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        if st.button("Generate Embedding"):
            with st.spinner("Processing..."):
                file_path = f"data/{uploaded_file.name}"

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                embedding_manager.add_csv(file_path, uploaded_file.name)
                # 🔥 Force reload embeddings
                st.session_state.embedding_version += 1

                # Clear cache so new embeddings load
                st.cache_resource.clear()

                st.success("Embedding created successfully!")
                st.rerun()

    st.subheader("📊 Uploaded Data Metadata")

    try:
        metadata = json.load(open("data/metadata.json"))
        if metadata:
            st.table(metadata)
        else:
            st.info("No data uploaded yet.")
    except:
        st.error("Unable to fetch metadata.")