import os
import faiss
import numpy as np
import sqlite3
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# 🔐 Gemini API Authentication
genai.configure(api_key="AIzaSyCqAZ_f8Lvn-i_jX2phis1ASD6rZWN25Q4")

# 💬 Load Gemini Model
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# 🔍 Load FAISS Index and Sentence Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')

faiss_index = faiss.read_index('bigbasket_faq_index.faiss')

# 📖 Load FAQ Knowledge Base (from Markdown)
with open('knowledge_Base1/BigBasketFAQ.md', 'r', encoding='utf-8') as f:
    faq_texts = f.read().split('\n\n')

# ⚙️ Structured Query Classifier
def is_structured_query(query):
    keywords = ['top', 'lowest', 'highest', 'price', 'under', 'above', 'brand', 'category']
    return any(word in query.lower() for word in keywords)

# 🔢 Semantic Search (FAISS + Embeddings)
def semantic_search(query, k=1):
    try:
        vector = model.encode([query]).astype('float32')
        D, I = faiss_index.search(vector, k)
        return [faq_texts[i] for i in I[0]]
    except Exception as e:
        return [f"FAISS search error: {str(e)}"]

# 🧾 Generate SQL using Gemini (with markdown cleanup)
def generate_sql_with_gemini(user_query):
    prompt = f"""
You are a data assistant for BigBasket. 
Generate a valid SQLite query for this user's request.

Table: bigbasket_data(
    product_id INTEGER, 
    product_name TEXT, 
    category TEXT, 
    sub_category TEXT, 
    brand TEXT, 
    sale_price REAL, 
    market_price REAL, 
    type TEXT, 
    rating REAL, 
    description TEXT
)

User question: "{user_query}"

Return only the SQL code. Do not explain.
"""
    response = gemini_model.generate_content(prompt)
    raw_sql = response.text.strip()
    if raw_sql.startswith("```"):
        raw_sql = raw_sql.strip("`")
        raw_sql = raw_sql.replace("sql", "", 1).strip()
    return raw_sql

# 🧮 Execute SQLite Query
def run_sql_query(query):
    try:
        conn = sqlite3.connect("bigbasket.db")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        return f"Error executing SQL: {e}"

# 🤖 Final Gemini Answer Generator
# 🤖 Final Gemini Answer Generator with system message
def summarize_with_gemini(user_query, context):
    prompt = f"""
You are a helpful assistant for BigBasket users.

If the retrieved data is not relevant to the question or seems incomplete, respond with a helpful message based on your general knowledge. You can also say things like "We couldn’t find an exact match, but here’s what might help..." or provide general tips related to the topic.

User asked: "{user_query}"

Here is the retrieved information:
{context}

Please provide a helpful and clear answer based on this.
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


# 🚀 Orchestration Function
def process_user_query(user_query):
    if is_structured_query(user_query):
        sql = generate_sql_with_gemini(user_query)
      #  print(f"🔧 Generated SQL: {sql}")
        result = run_sql_query(sql)
        print(result)

        context = pd.DataFrame(result).to_markdown(index=False) if isinstance(result, list) and result else "No results found or error occurred."
    else:
        matches = semantic_search(user_query)
        context = matches[0] if matches else "No matching FAQ found."

    final_response = summarize_with_gemini(user_query, context)
    return final_response

# 🧪 Debug Helpers
def check_db_connection():
    try:
        conn = sqlite3.connect("bigbasket.db")
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, conn)
        conn.close()
        return tables
    except Exception as e:
        return f"Error connecting to database: {e}"

def check_table_schema():
    try:
        conn = sqlite3.connect("bigbasket.db")
        query = "PRAGMA table_info(bigbasket_data);"
        schema = pd.read_sql_query(query, conn)
        conn.close()
        return schema
    except Exception as e:
        return f"Error fetching table schema: {e}"

def check_data_sample():
    try:
        conn = sqlite3.connect("bigbasket.db")
        query = "SELECT product_name, rating FROM bigbasket_data LIMIT 10;"
        sample_data = pd.read_sql_query(query, conn)
        conn.close()
        return sample_data
    except Exception as e:
        return f"Error fetching sample data: {e}"

def test_query():
    try:
        conn = sqlite3.connect("bigbasket.db")
        query = "SELECT product_name, rating FROM bigbasket_data ORDER BY rating DESC LIMIT 5;"
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        return f"Error executing query: {e}"

# 🔍 DB Integrity Check
print(check_db_connection())
print(check_table_schema())
print(check_data_sample())

# 🎯 CLI Interaction
if __name__ == "__main__":
    print("🤖 BigBasket AI Assistant Ready!")
    while True:
        query = input("\nAsk your question (or type 'exit'): ")
        if query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        response = process_user_query(query)
        print(f"\n💡 Assistant:\n{response}")
