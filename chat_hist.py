import streamlit as st
import pandas as pd
import faiss
import numpy as np
import sqlite3
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os

# 🔐 Gemini API Authentication
genai.configure(api_key="AIzaSyCqAZ_f8Lvn-i_jX2phis1ASD6rZWN25Q4")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ⚡ Efficient Data Loading (Only Once)
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')

if "faiss_index" not in st.session_state:
    if os.path.exists('bigbasket_faq_index.faiss'):
        st.session_state.faiss_index = faiss.read_index('bigbasket_faq_index.faiss')
    else:
        st.error("❌ FAISS index file not found.")

if "faq_texts" not in st.session_state:
    faq_path = 'knowledge_Base1/BigBasketFAQ.md'
    if os.path.exists(faq_path):
        with open(faq_path, 'r', encoding='utf-8') as f:
            st.session_state.faq_texts = f.read().split('\n\n')
    else:
        st.error("❌ FAQ file not found.")

# ⚙️ Utility Functions
def is_structured_query(query):
    keywords = ['top', 'lowest', 'highest', 'price', 'under', 'above', 'brand', 'category']
    return any(word in query.lower() for word in keywords)

def semantic_search(query, k=3):
    try:
        vector = st.session_state.model.encode([query]).astype('float32')
        D, I = st.session_state.faiss_index.search(vector, k)
        return [st.session_state.faq_texts[i] for i in I[0]]
    except Exception as e:
        return [f"FAISS search error: {str(e)}"]

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

def run_sql_query(query):
    try:
        conn = sqlite3.connect("bigbasket.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return f"Error executing SQL: {e}"

def summarize_with_gemini(user_query, context):
    prompt = f"""
You are a friendly and helpful assistant for BigBasket users.

Instructions:
1. If the user says something casual or friendly like "hi", "hello", or "hey", respond in a warm, polite way like:
   - "Hi there! 👋 How can I help you today?"
   - "Hello! 😊 What would you like to know about BigBasket?"

2. If the user's query matches relevant database or FAQ information, summarize and respond with that.

3. If the information retrieved isn't relevant or doesn't help, respond like this:
   "Sorry, I couldn't find an exact answer to your question. You can check BigBasket's website at https://www.bigbasket.com or reach out to their support team at customerservice@bigbasket.com."

User asked: "{user_query}"

Here is the retrieved information:
{context}

Please provide a helpful and clear answer based on this.
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# Store the query history
def store_query_history(user_query, assistant_response):
    try:
        conn = sqlite3.connect("bigbasket.db")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO query_history (user_query, assistant_response) 
            VALUES (?, ?)
        ''', (user_query, assistant_response))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error storing query history: {e}")

# Create the query history table if it doesn't exist
def create_history_table():
    try:
        conn = sqlite3.connect("bigbasket.db")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT,
                assistant_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error creating history table: {e}")

def process_user_query(user_query):
    if is_structured_query(user_query):
        sql = generate_sql_with_gemini(user_query)
        result = run_sql_query(sql)
        if isinstance(result, str):
            context = result
        else:
            context = result.to_markdown(index=False) if not result.empty else "No results found."
    else:
        matches = semantic_search(user_query)
        context = matches[0] if matches else "No matching FAQ found."

    assistant_response = summarize_with_gemini(user_query, context)
    
    # Store the query and response in the history
    store_query_history(user_query, assistant_response)
    
    return assistant_response

# 🎨 Streamlit UI
st.set_page_config(page_title="BigBasket AI Assistant", page_icon="🛒")
st.title("🤖 BigBasket AI Assistant")
st.markdown("Ask me anything about products, brands, prices, or policies!")

query = st.text_input("💬 Your Question:", placeholder="e.g., Show me top 5 rated snacks under ₹100")

if query:
    with st.spinner("Thinking..."):
        response = process_user_query(query)
        st.markdown("### 💡 Assistant Response:")
        st.markdown(response)

# Initialize the history table
create_history_table()
