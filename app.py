import os
import sqlite3
import pandas as pd
import streamlit as st
import chromadb
from openai import OpenAI

# ==========================================
# 1. Configuration & Setup
# ==========================================
st.set_page_config(page_title="NL2SQL Agent Demo", page_icon="🔍", layout="wide")

# The key is read from the environment (or a local .env file); never hard-code it.
_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not _api_key:
    st.error("DEEPSEEK_API_KEY is not set. Export it in your shell, then restart the app.")
    st.stop()

client = OpenAI(
    api_key=_api_key,
    base_url="https://api.deepseek.com"
)

DB_BASE_DIR = "data/database"

# Initialize ChromaDB for real RAG (Ensure your vector DB path is correct)
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    eval_collection = chroma_client.get_or_create_collection(name="spider_eval")
except Exception:
    eval_collection = None

# ==========================================
# 2. Dynamic Database Scanner
# ==========================================
@st.cache_data
def get_available_databases():
    """Scans the Spider dataset directory for all available databases."""
    if not os.path.exists(DB_BASE_DIR):
        return []
    return sorted([d for d in os.listdir(DB_BASE_DIR) if os.path.isdir(os.path.join(DB_BASE_DIR, d))])

# ==========================================
# 3. Agent Core Functions
# ==========================================
def get_real_schema(db_path):
    """Fallback: Directly extract full schema from the SQLite file for the UI Demo"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        conn.close()
        return "\n".join([t[0] for t in tables if t[0]])
    except Exception as e:
        return f"Error extracting schema: {str(e)}"

def execute_sql_for_dataframe(db_path, sql_query):
    """Executes SQL and returns a Pandas DataFrame."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return True, df
    except Exception as e:
        return False, str(e)

def generate_and_correct(query, db_path, db_id, max_retries=3):
    """The Complete Agent Loop with Real RAG."""
    # Dynamically fetch schema based on the selected database
    schema = get_real_schema(db_path)

    prompt = f"""Convert the user's question into a SQL query.
    [Schema]: {schema}
    [Question]: {query}
    Output ONLY raw SQL. No markdown, no explanations."""

    messages = [
        {"role": "system", "content": "You are a highly accurate Text-to-SQL assistant."},
        {"role": "user", "content": prompt}
    ]

    last_error = ""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat", messages=messages, temperature=0.0
            )
            gen_sql = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()

            success, exec_res = execute_sql_for_dataframe(db_path, gen_sql)

            if success:
                return gen_sql, exec_res, f"Success on attempt {attempt + 1}"
            else:
                last_error = exec_res
                messages.append({"role": "assistant", "content": gen_sql})
                messages.append({"role": "user", "content": f"Database error: '{last_error}'. Please reflect and fix the SQL. Output ONLY raw SQL."})

        except Exception as api_error:
            return None, None, f"API Error: {str(api_error)}"

    return None, None, f"Failed after {max_retries} retries. Last error: {last_error}"


# ==========================================
# 3b. Risk-gated agent (new layer)
# ==========================================
@st.cache_resource
def load_risk_agent():
    """Load the Bayesian decision stack once per session.

    Returns ``None`` when the trained models are missing, in which case the
    page falls back to the original loop and says so.
    """
    from config import MODEL_DIR
    model_file = os.path.join(MODEL_DIR, "lr_risk_model.joblib")
    if not os.path.isfile(model_file):
        return None
    try:
        from bayesian_agent import BayesianNL2SQLAgent
        from error_classifier import SQLErrorClassifier
        from llm_client import DeepSeekClient

        models = SQLErrorClassifier.load(MODEL_DIR)
        return BayesianNL2SQLAgent(client=DeepSeekClient(), classifier=models,
                                   model_key="LR")
    except Exception as exc:          # missing key, broken model, ...
        st.sidebar.warning(f"Risk layer disabled: {type(exc).__name__}")
        return None


def analyse_query(query, db_path, db_id):
    """Run the risk-gated agent; return a uniform dict for the UI."""
    agent = load_risk_agent()
    if agent is None:
        sql, df, status = generate_and_correct(query, db_path, db_id)
        return {"status": status, "sql": sql, "data": df, "p_wrong": None,
                "trace": "legacy loop", "calls": None}

    res = agent.run(query, db_id, db_path)
    ok, data = (False, None)
    if res.status == "answered" and res.final_sql:
        ok, data = execute_sql_for_dataframe(db_path, res.final_sql)
    return {
        "status": ("Success" if ok else ("Clarification required"
                                         if res.status == "clarify"
                                         else "Execution failed")),
        "sql": res.final_sql,
        "data": data if ok else None,
        "p_wrong": res.p_wrong_final,
        "trace": " -> ".join(res.action_path),
        "calls": res.n_calls,
        "error": res.error,
    }

# ==========================================
# 4. Streamlit UI Layout
# ==========================================
# Sidebar Configuration
st.sidebar.title("⚙️ System Config")
db_list = get_available_databases()

if not db_list:
    st.sidebar.error(f"Could not find databases in '{DB_BASE_DIR}'. Please check your data folder.")
    selected_db = None
else:
    selected_db = st.sidebar.selectbox("🎯 Target Database:", db_list)
    current_db_path = os.path.join(DB_BASE_DIR, selected_db, f"{selected_db}.sqlite")
    st.sidebar.success(f"Connected to: {selected_db}.sqlite")

# Main Content
st.title("🤖 Intelligent Text-to-SQL Agent")
st.markdown("Select a database from the sidebar, then enter a natural language query below.")

if selected_db:
    # User Input
    user_query = st.text_input("Ask a question about your data:", placeholder="e.g., Find the names of all employees...")

    with st.expander("👁️ View Injected Schema (Debug)"):
        st.code(get_real_schema(current_db_path), language="sql")

    if st.button("Generate & Execute", type="primary"):
        if not user_query:
            st.warning("Please enter a question first.")
        else:
            with st.spinner(f"Agent is reasoning over '{selected_db}' database..."):
                out = analyse_query(user_query, current_db_path, selected_db)
                st.divider()

                # --- decision trace (the risk layer's explainability) -----
                c1, c2, c3 = st.columns(3)
                c1.metric("Risk P(wrong)",
                          "n/a" if out["p_wrong"] is None
                          else f"{out['p_wrong']:.2f}")
                c2.metric("LLM calls",
                          "-" if out["calls"] is None else out["calls"])
                c3.metric("Decision", out["trace"].split(" -> ")[-1][:24])

                if out["status"] == "Clarification required":
                    st.warning(
                        "The risk model considers this question too ambiguous to "
                        "answer safely. Please add a filter, a time range or the "
                        "metric you mean.")
                    with st.expander("👁️ Decision trace"):
                        st.code(out["trace"], language="text")

                elif out["status"].startswith("Success"):
                    st.success(f"Agent Status: {out['status']}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("💻 Final Generated SQL")
                        st.code(out["sql"], language="sql")
                        with st.expander("👁️ Decision trace"):
                            st.code(out["trace"], language="text")

                    with col2:
                        st.subheader("📊 Execution Result")
                        if isinstance(out["data"], pd.DataFrame):
                            if out["data"].empty:
                                st.warning("Query executed successfully, but returned 0 rows.")
                            else:
                                st.dataframe(out["data"], use_container_width=True)
                else:
                    st.error("Agent failed to generate a valid SQL query.")
                    st.code(out.get("error") or out["status"], language="bash")