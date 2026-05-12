import sqlite3
import os
from openai import OpenAI

# 1. Configure DeepSeek API Key
# Replace the string below with your actual DeepSeek API Key (starts with 'sk-')
os.environ["DEEPSEEK_API_KEY"] = "sk-307594d3f334466d94ec9088e76d5f96"

# 2. Initialize the client using DeepSeek's base URL
# DeepSeek is fully compatible with the OpenAI Python SDK
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def get_table_schema(db_path):
    """
    Connect to the SQLite database and extract the DDL (Data Definition Language) for all tables.
    In the Baseline phase, we feed the entire schema to the LLM.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query the SQLite master table to get all 'CREATE TABLE' statements
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schemas = cursor.fetchall()
    conn.close()

    # Concatenate the extracted DDL statements into a single string
    return "\n".join([schema[0] for schema in schemas if schema[0]])

def generate_sql_baseline(question, schema_text):
    """
    Invoke the LLM to translate a natural language query into an SQL statement.
    """
    prompt = f"""
    You are a senior database architect. Based on the provided database schema, 
    convert the user's natural language question into a valid, executable SQL query.
    
    [Database Schema]:
    {schema_text}
    
    [User Question]:
    {question}
    
    Please output ONLY the raw SQL query. Do not include any explanations, formatting, or markdown tags (such as ```sql).
    """

    response = client.chat.completions.create(
        model="deepseek-chat", # Use 'deepseek-chat' for general tasks including text-to-SQL
        messages=[
            {"role": "system", "content": "You are a highly accurate Text-to-SQL assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0 # Set temperature to 0 for maximum determinism in code generation
    )

    return response.choices[0].message.content.strip()

# ================= Test Execution Area =================
if __name__ == "__main__":
    # Ensure this path matches the SQLite file in your 'data' directory
    db_file_path = "data/school.sqlite"

    # Simulated natural language query (in English to match the environment)
    user_query = "Find the names of all students who are older than 18 and belong to the 'Computer Science' department."

    print("Extracting database schema...")
    try:
        schema_info = get_table_schema(db_file_path)
        print("Schema extracted successfully! Calling DeepSeek API to generate SQL...")

        generated_sql = generate_sql_baseline(user_query, schema_info)

        print("\n" + "="*50)
        print("[Generated SQL by DeepSeek]:")
        print(generated_sql)
        print("="*50 + "\n")

    except Exception as e:
        print(f"Execution failed. Please verify the database path and API configuration. Error details: {e}")