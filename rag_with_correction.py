import sqlite3
import os
import chromadb
from openai import OpenAI

# 1. Configuration
# REMEMBER TO PASTE YOUR REAL API KEY HERE
os.environ["DEEPSEEK_API_KEY"] = "sk-307594d3f334466d94ec9088e76d5f96"
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 2. Initialize ChromaDB Vector Store
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="schema_collection")

def retrieve_relevant_schema(query, top_k=2):
    """Retrieve the most relevant table schemas using RAG."""
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved_ddls = "\n".join(results['documents'][0])
    return retrieved_ddls

def execute_sql_in_sandbox(db_path, sql_query):
    """
    Attempt to execute the generated SQL in the SQLite database.
    Returns (True, result_data) if successful.
    Returns (False, error_message) if it fails.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)

def generate_and_correct_sql(query, db_path, max_retries=3):
    """
    Core Agent Loop: Generate SQL -> Execute -> Catch Error -> Reflect & Correct
    """
    retrieved_schema = retrieve_relevant_schema(query)

    # Initial Prompt
    prompt = f"""
    You are a senior database architect. Convert the user's natural language question into a valid SQL query.
    
    [Relevant Database Schema]:
    {retrieved_schema}
    
    [User Question]:
    {query}
    
    Output ONLY the raw SQL query. No markdown tags, no explanations.
    """

    messages = [
        {"role": "system", "content": "You are a highly accurate Text-to-SQL assistant."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(1, max_retries + 1):
        print(f"\n--- [Attempt {attempt}/{max_retries}] Generating SQL... ---")

        # 1. Generate SQL
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1
        )
        generated_sql = response.choices[0].message.content.strip()

        # Clean up potential markdown formatting from LLM
        generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
        print(f"[Generated SQL]: {generated_sql}")

        # 2. Execute in Sandbox
        print("[Sandbox Action]: Executing SQL in database...")
        success, execution_result = execute_sql_in_sandbox(db_path, generated_sql)

        # 3. Check Result and Reflect
        if success:
            print("[Success]: SQL executed perfectly without syntax errors!")
            return generated_sql, execution_result
        else:
            error_msg = execution_result
            print(f"[Error Caught]: {error_msg}")

            if attempt < max_retries:
                print("[Self-Correction Action]: Sending error log back to LLM for reflection...")
                # Append the failed attempt and the database error to the message history
                messages.append({"role": "assistant", "content": generated_sql})
                reflection_prompt = f"Executing the above SQL resulted in this database error: '{error_msg}'. Please analyze the error and provide a corrected SQL query. Output ONLY the raw SQL."
                messages.append({"role": "user", "content": reflection_prompt})
            else:
                print("[Failed]: Maximum retries reached. Unable to generate correct SQL.")
                return None, error_msg

# ================= Test Execution Area =================
if __name__ == "__main__":
    db_file_path = "data/school.sqlite"

    print("Welcome to the NL2SQL Agent (RAG + Self-Correction)")
    while True:
        user_query = input("\nUser Query: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        final_sql, data = generate_and_correct_sql(user_query, db_file_path)

        if final_sql:
            print("\n" + "="*50)
            print("🎉 [Final Approved SQL]:")
            print(final_sql)
            print(f"📊 [Sample Data Retrieved]: {data[:3]} ...")
            print("="*50 + "\n")