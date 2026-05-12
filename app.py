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

client = OpenAI(
    api_key="sk-459ed873d35842abbaa87dd1b1d5c15b",
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
                sql, result_data, status = generate_and_correct(user_query, current_db_path, selected_db)

                st.divider()

                if "Success" in status:
                    st.success(f"Agent Status: {status}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("💻 Final Generated SQL")
                        st.code(sql, language="sql")

                    with col2:
                        st.subheader("📊 Execution Result")
                        if isinstance(result_data, pd.DataFrame):
                            if result_data.empty:
                                st.warning("Query executed successfully, but returned 0 rows.")
                            else:
                                st.dataframe(result_data, use_container_width=True)
                else:
                    st.error("Agent failed to generate a valid SQL query.")
                    st.code(status, language="bash")