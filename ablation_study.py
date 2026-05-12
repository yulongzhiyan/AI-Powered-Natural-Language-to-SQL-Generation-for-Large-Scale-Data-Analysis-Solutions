import json
import sqlite3
import os
import csv
from openai import OpenAI
import chromadb

# ================= 1. Configuration =================
os.environ["DEEPSEEK_API_KEY"] = "sk-307594d3f334466d94ec9088e76d5f96" # 记得填写真实的 Key
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

SPIDER_DEV_PATH = "data/dev.json"
DATABASE_DIR = "data/database"

# ！！！消融实验核心开关 ！！！
# 请选择你要跑的基线模式: "LLM_ONLY" 或 "LLM_RAG"
ABLATION_MODE = "LLM_RAG"

# 根据模式自动设置输出的 CSV 文件名
CSV_OUT_PATH = f"ablation_results_{ABLATION_MODE.lower()}.csv"

# 初始化向量库 (仅在 LLM_RAG 模式下使用)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
eval_collection = chroma_client.get_or_create_collection(name="spider_eval")

# ================= 2. Core Functions =================

def execute_sql(db_path, sql_query):
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 3000")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        res = cursor.fetchall()
        conn.close()
        return True, res
    except Exception as e:
        return False, str(e)

def get_full_schema_llm_only(db_path):
    """【基线1专用】直接暴力提取目标数据库的所有表结构，无 RAG 过滤"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    conn.close()
    return "\n".join([t[0] for t in tables if t[0]])

def retrieve_schema_rag(query, db_id, top_k=3):
    """【基线2专用】使用向量检索，只提取 Top-K 表结构"""
    results = eval_collection.query(
        query_texts=[query], n_results=top_k, where={"db_id": db_id}
    )
    if results['documents'] and results['documents'][0]:
        return "\n".join(results['documents'][0])
    return ""

def generate_sql_one_shot(query, db_path, db_id):
    """
    消融实验专用 Agent：
    1. 根据 ABLATION_MODE 决定如何获取 Schema
    2. 只有 1 次生成机会 (无 max_retries，无错误反思)
    """
    if ABLATION_MODE == "LLM_ONLY":
        schema = get_full_schema_llm_only(db_path)
    elif ABLATION_MODE == "LLM_RAG":
        schema = retrieve_schema_rag(query, db_id)

    prompt = f"""Convert the user's question into a SQL query.
    [Schema]: {schema}
    [Question]: {query}
    Output ONLY raw SQL. No markdown, no explanations."""

    messages = [
        {"role": "system", "content": "You are a highly accurate Text-to-SQL assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.0
        )
        gen_sql = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
        success, exec_res = execute_sql(db_path, gen_sql)

        status = "Success" if success else exec_res
        return gen_sql, exec_res if success else [], status
    except Exception as e:
        return None, [], f"API Error: {str(e)}"

# ================= 3. Evaluation Pipeline =================

def run_ablation(limit=None):
    with open(SPIDER_DEV_PATH, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    if limit: dev_data = dev_data[:limit]

    total = len(dev_data)
    correct_count = 0

    file_exists = os.path.isfile(CSV_OUT_PATH)
    with open(CSV_OUT_PATH, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Test_ID', 'Database', 'Question', 'Is_Correct', 'Error_Log'])
        if not file_exists: writer.writeheader()

        print(f"\n🚀 启动消融实验: 模式 [{ABLATION_MODE}]")
        print(f"📁 结果将保存至: {CSV_OUT_PATH}\n")

        for i, item in enumerate(dev_data):
            db_id = item['db_id']
            question = item['question']
            gold_sql = item['query']
            db_path = os.path.join(DATABASE_DIR, db_id, f"{db_id}.sqlite")

            # Gold execution
            gold_success, gold_res = execute_sql(db_path, gold_sql)

            # AI execution (One-shot, no correction)
            gen_sql, gen_res, status_msg = generate_sql_one_shot(question, db_path, db_id)

            is_correct = False
            error_log = status_msg

            if gold_success and gen_sql:
                try:
                    if sorted(gold_res) == sorted(gen_res):
                        is_correct = True
                        correct_count += 1
                        error_log = "Matched"
                except TypeError:
                    error_log = "Type/Order Mismatch"

            writer.writerow({
                'Test_ID': i + 1, 'Database': db_id, 'Question': question,
                'Is_Correct': is_correct, 'Error_Log': error_log
            })
            csv_file.flush()

            # 简单进度打印
            if (i+1) % 50 == 0 or (i+1) == total:
                print(f"进度: {i+1}/{total} | 当前准确率: {(correct_count/(i+1))*100:.2f}%")

    accuracy = (correct_count / total) * 100
    print("\n" + "="*50)
    print(f"🏆 消融实验 [{ABLATION_MODE}] 跑分完成!")
    print(f"📊 最终准确率: {accuracy:.2f}% ({correct_count}/{total})")
    print("="*50)

if __name__ == "__main__":
    # 正式跑全量请设为 limit=None
    run_ablation(limit=None)