import json
import sqlite3
import os
from openai import OpenAI
import chromadb

# ================= Configuration =================
os.environ["DEEPSEEK_API_KEY"] = "sk-307594d3f334466d94ec9088e76d5f96" # REPLACE THIS!
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

SPIDER_DEV_PATH = "data/dev.json"
DATABASE_DIR = "data/database"

# Initialize Vector DB (using a dedicated collection for evaluation)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
eval_collection = chroma_client.get_or_create_collection(name="spider_eval")

# ================= Helper Functions =================

def execute_sql(db_path, sql_query):
    """Executes SQL and returns the fetched data or the error message."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        res = cursor.fetchall()
        conn.close()
        return True, res
    except Exception as e:
        return False, str(e)

def index_database_schema(db_path, db_id):
    """Dynamically indexes a specific Spider database into ChromaDB."""
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
    """Retrieve schema only for the specific db_id being queried."""
    results = eval_collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"db_id": db_id} # Filter by current database
    )
    if results['documents'] and results['documents'][0]:
        return "\n".join(results['documents'][0])
    return ""

def generate_and_correct(query, db_path, db_id, max_retries=3):
    """The Agent Loop for evaluation."""
    schema = retrieve_schema(query, db_id)
    if not schema:
        index_database_schema(db_path, db_id)
        schema = retrieve_schema(query, db_id)

    prompt = f"""Convert the user's question into a SQL query.
    [Schema]: {schema}
    [Question]: {query}
    Output ONLY raw SQL."""

    messages = [
        {"role": "system", "content": "You are a precise Text-to-SQL assistant."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.0
        )
        gen_sql = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()

        success, exec_res = execute_sql(db_path, gen_sql)
        if success:
            return gen_sql, exec_res
        else:
            messages.append({"role": "assistant", "content": gen_sql})
            messages.append({"role": "user", "content": f"Error: '{exec_res}'. Fix the SQL. Output ONLY raw SQL."})

    return None, "Max retries failed."

# ================= Evaluation Pipeline =================

def run_evaluation(limit=5):
    """
    Run evaluation on a subset of the dataset to verify logic.
    Set limit=None to run all 1034 queries (takes time and tokens!).
    """
    print(f"Loading Spider Dev Set... (Testing {limit if limit else 'ALL'} queries)")
    with open(SPIDER_DEV_PATH, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    if limit:
        dev_data = dev_data[:limit]

    correct_count = 0
    total = len(dev_data)

    for i, item in enumerate(dev_data):
        db_id = item['db_id']
        question = item['question']
        gold_sql = item['query']
        db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")

        print(f"\n--- [Test {i+1}/{total}] DB: {db_id} ---")
        print(f"Q: {question}")

        # 1. Get Gold Result
        gold_success, gold_res = execute_sql(db_path, gold_sql)

        # 2. Get AI Generated Result (with RAG & Correction)
        gen_sql, gen_res = generate_and_correct(question, db_path, db_id)

        # 3. Compare Execution Results
        is_correct = False
        if gold_success and gen_sql:
            # Sort results to ensure order doesn't cause false negatives unless ORDER BY is strictly required
            if sorted(gold_res) == sorted(gen_res):
                is_correct = True

        if is_correct:
            correct_count += 1
            print("✅ Status: CORRECT")
        else:
            print("❌ Status: WRONG")
            print(f"   Gold SQL: {gold_sql}")
            print(f"   Gen SQL:  {gen_sql}")

    accuracy = (correct_count / total) * 100
    print("\n" + "="*40)
    print(f"🏆 Evaluation Complete!")
    print(f"📊 Execution Accuracy (EX): {accuracy:.2f}% ({correct_count}/{total})")
    print("="*40)

if __name__ == "__main__":
    # We start by testing just the first 5 questions to ensure no path/file errors.
    run_evaluation(limit=5)