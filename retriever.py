import os
import faiss
import numpy as np
import sqlite3
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# 🔐 Authenticate Gemini
#genai.configure(api_key=os.getenv("AIzaSyBECSLUqFEwRMMG2cOnhAceEZKKZYzcS5A"))  # Or replace with your actual API key

# 💬 Initialize Gemini Model
#gemini_model = genai.GenerativeModel("gemini-1.5-pro")




genai.configure(api_key="AIzaSyCqAZ_f8Lvn-i_jX2phis1ASD6rZWN25Q4")


gemini_model = genai.GenerativeModel("gemini-1.5-pro")


# 🔍 Load FAISS Index and Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index('bigbasket_faq_index.faiss')

# ✅ Read FAQ texts from Markdown (instead of pkl)
with open('knowledge_Base1/BigBasketFAQ.md', 'r', encoding='utf-8') as f:
    faq_texts = f.read().split('\n\n')

# 🧠 Query Type Detection
def is_structured_query(query):
    keywords = ['top', 'lowest', 'highest', 'price', 'under', 'above', 'brand', 'category']
    return any(word in query.lower() for word in keywords)

# 🔢 Semantic Search using FAISS
def semantic_search(query, k=1):
    try:
        vector = model.encode([query]).astype('float32')
        D, I = faiss_index.search(vector, k)
        return [faq_texts[i] for i in I[0]]
    except Exception as e:
        return [f"FAISS search error: {str(e)}"]

# 🧾 Generate SQL using Gemini
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
    return response.text.strip()

# 🧮 Execute SQL Query
def run_sql_query(query):
    try:
        conn = sqlite3.connect("bigbasket.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return f"Error executing SQL: {e}"

# 🧠 Final Answer using Gemini
def summarize_with_gemini(user_query, context):
    prompt = f"""
User asked: "{user_query}"

Here is the retrieved information:
{context}

Please provide a helpful and clear answer.
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# 🚀 Main Function
def process_user_query(user_query):
    if is_structured_query(user_query):
        sql = generate_sql_with_gemini(user_query)
        print(f"🔧 Generated SQL: {sql}")
        result = run_sql_query(sql)
        if isinstance(result, pd.DataFrame) and not result.empty:
            context = result.to_markdown(index=False)
        else:
            context = "No results found or error occurred."
    else:
        matches = semantic_search(user_query)
        context = matches[0] if matches else "No matching FAQ found."

    final_response = summarize_with_gemini(user_query, context)
    return final_response

# 🧪 Interactive CLI
if __name__ == "__main__":
    print("🤖 BigBasket AI Assistant Ready!")
    while True:
        query = input("\nAsk your question (or type 'exit'): ")
        if query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        response = process_user_query(query)
        print(f"\n💡 Assistant:\n{response}")
