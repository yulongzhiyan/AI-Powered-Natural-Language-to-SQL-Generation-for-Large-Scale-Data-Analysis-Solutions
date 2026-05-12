import sqlite3
import os
import chromadb
from openai import OpenAI

# 1. Configuration (Keep using your DeepSeek key)
os.environ["DEEPSEEK_API_KEY"] = "sk-307594d3f334466d94ec9088e76d5f96"
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 2. Initialize ChromaDB Vector Store
# This will create an embedding database in a local folder named 'chroma_db'
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Using the default lightweight embedding function from Chroma
collection = chroma_client.get_or_create_collection(name="schema_collection")

def index_schema_to_vectordb(db_path):
    """
    Step 1: Extract tables from SQLite and vectorize them into ChromaDB.
    This simulates indexing a massive database for efficient retrieval.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    conn.close()

    docs = []
    metadata = []
    ids = []

    for idx, (table_name, create_sql) in enumerate(tables):
        # We index the raw SQL creation statement to capture table intent
        docs.append(create_sql)
        metadata.append({"table_name": table_name})
        ids.append(f"table_{idx}")

    # Check if collection is empty before indexing to avoid duplicate entries
    if collection.count() == 0:
        print(f"Indexing {len(tables)} tables into vector database...")
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
        print("Indexing complete.")
    else:
        print("Vector database already contains indexed schemas.")

def retrieve_relevant_schema(query, top_k=2):
    """
    Step 2: RAG Retrieval - Find the top_k most relevant tables based on the user's query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    # Concatenate the retrieved DDLs
    retrieved_ddls = "\n".join(results['documents'][0])
    retrieved_tables = [meta['table_name'] for meta in results['metadatas'][0]]

    return retrieved_ddls, retrieved_tables

def generate_sql_with_rag(query, db_path):
    """
    Step 3: Dynamic Prompting - Generate SQL using ONLY the retrieved schema.
    """
    # First, make sure our vector DB is populated
    index_schema_to_vectordb(db_path)

    # Retrieve only relevant schema context
    retrieved_schema, table_names = retrieve_relevant_schema(query, top_k=2)

    print(f"\n[RAG Action]: Extracted Top-K relevant tables: {table_names}")

    prompt = f"""
    You are a senior database architect. Based on the provided relevant database schema, 
    convert the user's natural language question into a valid, executable SQL query.
    
    [Relevant Database Schema (Retrieved context)]:
    {retrieved_schema}
    
    [User Question]:
    {query}
    
    Please output ONLY the raw SQL query. Do not include any explanations, formatting, or markdown tags.
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a highly accurate Text-to-SQL assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content.strip()

# ================= Test Execution Area =================
if __name__ == "__main__":
    db_file_path = "data/school.sqlite"

    # Let's use an interactive prompt so you don't have to hardcode the query anymore!
    print("Welcome to the NL2SQL RAG System (Type 'exit' to quit)")
    while True:
        user_query = input("\nUser Query: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        try:
            generated_sql = generate_sql_with_rag(user_query, db_file_path)

            print("\n" + "="*50)
            print("[Generated SQL]:")
            print(generated_sql)
            print("="*50 + "\n")

        except Exception as e:
            print(f"Error during execution: {e}")