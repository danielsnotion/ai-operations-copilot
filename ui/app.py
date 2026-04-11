from dotenv import load_dotenv
import streamlit as st
import requests
import uuid
import time
import os

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
st.set_page_config(layout="wide")
st.title("🤖 AI Operations Copilot")

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

# ---------------- SESSION STATE ---------------- #
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

# ---------------- CLEAN CSS ---------------- #
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
def stream_text(text, delay=0.01):
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

    # -------- APPLY PENDING RESPONSE -------- #
    if st.session_state.pending_response:
        st.session_state.messages.append(st.session_state.pending_response)
        st.session_state.pending_response = None

    # -------- CHAT HISTORY -------- #
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("latency") is not None:
                st.caption(f"⚡ Response time: {msg['latency']:.2f} sec")

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
                            requests.post(f"{BACKEND_URL}/feedback", json={
                                "query": msg["query"],
                                "response": msg["content"],
                                "feedback": "positive"
                            })
                            msg["feedback"] = "positive"
                            st.session_state.feedback_status[msg["id"]] = "positive"
                            st.rerun()

                    with col2:
                        if st.button("👎 Not Helpful", key=f"down_{msg['id']}"):
                            requests.post(f"{BACKEND_URL}/feedback", json={
                                "query": msg["query"],
                                "response": msg["content"],
                                "feedback": "negative"
                            })
                            msg["feedback"] = "negative"
                            st.session_state.feedback_status[msg["id"]] = "negative"
                            st.rerun()

                elif temp_feedback:
                    if temp_feedback == "positive":
                        st.success("👍 Feedback saved!")
                    else:
                        st.warning("👎 Feedback saved!")
                    del st.session_state.feedback_status[msg["id"]]

                elif feedback:
                    st.caption(f"Feedback: {feedback}")

    # -------- STREAMING (ABOVE INPUT) -------- #
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

    # -------- INPUT (ALWAYS VISIBLE) -------- #
    query = st.chat_input(
        "Ask your question...",
        disabled=st.session_state.streaming
    )

    # -------- HANDLE QUERY -------- #
    if query:

        if not st.session_state.auth_mode:
            st.warning("⚠️ Please login or provide API key first")
            st.stop()

        # Show user immediately
        st.session_state.messages.append({
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": query
        })

        auto_scroll()

        try:
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={
                    "query": query,
                    "llm_model": llm_model,
                    "framework": framework,
                    "api_key": st.session_state.api_key,
                    "auth_mode": st.session_state.auth_mode
                }
            ).json()

            api_payload = response.get("response", {})

            if isinstance(api_payload, dict):
                answer = api_payload.get("final_answer", "")
                trace = api_payload.get("trace", [])
            else:
                answer = api_payload
                trace = []

            latency = response.get("latency")

        except Exception as e:
            answer = f"Backend error: {str(e)}"
            trace = []
            latency = None

        # Store for streaming
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
                try:
                    requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": uploaded_file}
                    )
                    st.success("Embedding created successfully!")
                except:
                    st.error("Failed to generate embeddings")

    st.subheader("📊 Uploaded Data Metadata")

    try:
        metadata = requests.get(f"{BACKEND_URL}/metadata").json()
        if metadata:
            st.table(metadata)
        else:
            st.info("No data uploaded yet.")
    except:
        st.error("Unable to fetch metadata.")