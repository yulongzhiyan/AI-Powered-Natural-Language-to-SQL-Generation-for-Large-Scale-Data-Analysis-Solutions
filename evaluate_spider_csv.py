import json
import sqlite3
import os
import csv
from openai import OpenAI
import chromadb

# ================= 1. Configuration =================
# REMEMBER TO PASTE YOUR REAL API KEY HERE
os.environ["DEEPSEEK_API_KEY"] = "sk-307594d3f334466d94ec9088e76d5f96"
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

SPIDER_DEV_PATH = "data/dev.json"
DATABASE_DIR = "data/database"
CSV_OUT_PATH = "spider_evaluation_results.csv" # The file where your thesis data will be saved

# Initialize Vector DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
eval_collection = chroma_client.get_or_create_collection(name="spider_eval")

# ================= 2. Core Functions (RAG + Correction) =================

def execute_sql(db_path, sql_query):
    """Executes SQL in SQLite and returns fetched data or error message."""
    try:
        conn = sqlite3.connect(db_path)
        # Timeout added to prevent infinite loops on bad queries
        conn.execute("PRAGMA busy_timeout = 3000")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        res = cursor.fetchall()
        conn.close()
        return True, res
    except Exception as e:
        return False, str(e)

def index_database_schema(db_path, db_id):
    """Dynamically indexes a specific database schema into ChromaDB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    conn.close()

    docs, metadata, ids = [], [], []
    for idx, (table_name, create_sql) in enumerate(tables):
        if create_sql:
            docs.append(create_sql)
            metadata.append({"table_name": table_name, "db_id": db_id})
            ids.append(f"{db_id}_{table_name}_{idx}")

    if docs:
        eval_collection.add(documents=docs, metadatas=metadata, ids=ids)

def retrieve_schema(query, db_id, top_k=3):
    """Retrieve top_k schemas only for the target database."""
    results = eval_collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"db_id": db_id}
    )
    if results['documents'] and results['documents'][0]:
        return "\n".join(results['documents'][0])
    return ""

def generate_and_correct(query, db_path, db_id, max_retries=3):
    """The Agent Loop: RAG -> Generate -> Sandbox Test -> Self-Correct."""
    schema = retrieve_schema(query, db_id)
    if not schema:
        index_database_schema(db_path, db_id)
        schema = retrieve_schema(query, db_id)

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

            success, exec_res = execute_sql(db_path, gen_sql)
            if success:
                return gen_sql, exec_res, "Success"
            else:
                last_error = exec_res
                messages.append({"role": "assistant", "content": gen_sql})
                messages.append({"role": "user", "content": f"Database error: '{last_error}'. Please reflect and fix the SQL. Output ONLY raw SQL."})
        except Exception as api_error:
            return None, None, f"API Error: {str(api_error)}"

    return None, None, f"Failed after {max_retries} retries. Last error: {last_error}"

# ================= 3. Evaluation Pipeline with CSV Logging =================

def run_evaluation_with_logging(limit=None):
    """
    Run the benchmark and save results in real-time to a CSV file.
    """
    with open(SPIDER_DEV_PATH, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    if limit:
        dev_data = dev_data[:limit]

    total = len(dev_data)
    correct_count = 0

    # Initialize CSV File and write header
    file_exists = os.path.isfile(CSV_OUT_PATH)
    with open(CSV_OUT_PATH, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Test_ID', 'Database', 'Question', 'Gold_SQL', 'Generated_SQL', 'Is_Correct', 'Error_Log']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header only if the file is newly created
        if not file_exists:
            writer.writeheader()

        print(f"\n🚀 Starting Evaluation Pipeline... (Testing {total} queries)")
        print(f"📁 Results will be saved in real-time to: {CSV_OUT_PATH}\n")

        for i, item in enumerate(dev_data):
            db_id = item['db_id']
            question = item['question']
            gold_sql = item['query']
            db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")

            print(f"[{i+1}/{total}] DB: {db_id} | Q: {question}")

            # Get Gold Result
            gold_success, gold_res = execute_sql(db_path, gold_sql)

            # Get AI Result
            gen_sql, gen_res, status_msg = generate_and_correct(question, db_path, db_id)

            # Compare
            is_correct = False
            error_log = status_msg

            if gold_success and gen_sql:
                try:
                    # Attempt to sort and compare
                    if sorted(gold_res) == sorted(gen_res):
                        is_correct = True
                        correct_count += 1
                        error_log = "None (Matched)"
                        print("  👉 Result: ✅ CORRECT")
                    else:
                        error_log = "Data Mismatch (Executed but data differs)"
                        print("  👉 Result: ❌ DATA MISMATCH")
                except TypeError as e:
                    # Catch the exact error you just experienced!
                    error_log = "Type/Order Mismatch (AI selected wrong column types or order)"
                    print("  👉 Result: ❌ TYPE/ORDER MISMATCH")
            else:
                print(f"  👉 Result: ❌ ERROR ({status_msg})")

            # Write to CSV immediately
            writer.writerow({
                'Test_ID': i + 1,
                'Database': db_id,
                'Question': question,
                'Gold_SQL': gold_sql,
                'Generated_SQL': gen_sql if gen_sql else "N/A",
                'Is_Correct': is_correct,
                'Error_Log': error_log
            })

            # Force flush to disk to prevent data loss
            csv_file.flush()

    accuracy = (correct_count / total) * 100
    print("\n" + "="*50)
    print(f"🏆 FULL EVALUATION COMPLETE!")
    print(f"📊 Final Execution Accuracy (EX): {accuracy:.2f}% ({correct_count}/{total})")
    print(f"💾 Detailed logs saved to: {CSV_OUT_PATH}")
    print("="*50)

if __name__ == "__main__":
    # ⚠️ IMPORTANT: Set limit=10 for a quick test of the CSV generation.
    # Change to limit=None when you are ready to run all 1034 queries!
    run_evaluation_with_logging(limit=None)