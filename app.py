import streamlit as st
import pandas as pd
import faiss
import numpy as np
import sqlite3
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os

# 🔐 Gemini API Authentication
genai.configure(api_key="AIzaSyC-O2kw8mkhPl5BdHxFAPgEglXMsRAX6WM")
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# ⚡ Efficient Data Loading (Only Once)
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')

if "faiss_index" not in st.session_state:
    if os.path.exists('bigbasket_faq_index.faiss'):
        st.session_state.faiss_index = faiss.read_index('bigbasket_faq_index.faiss')
    else:
        st.error(" FAISS index file not found.")

if "faq_texts" not in st.session_state:
    faq_path = 'knowledge_Base1/BigBasketFAQ.md'
    if os.path.exists(faq_path):
        with open(faq_path, 'r', encoding='utf-8') as f:
            st.session_state.faq_texts = f.read().split('\n\n')
    else:
        st.error(" FAQ file not found.")

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
    # Custom response for general help questions
    general_help_phrases = ["hi", "hello", "hey", "how are you", "what's up","how can you help", "what can you do", "help me", "assist me"]
    if any(phrase in user_query.lower() for phrase in general_help_phrases):
        return (
            "Hi there! 😊 How can I assist you today with your shopping or queries?"
            "I can help you with anything related to your BigBasket experience — "
            "from finding the right products, exploring deals, tracking your order, "
            "to managing your account settings. Need help with something specific like "
            "changing your delivery address or checking available delivery slots? Just ask!"
        )

    prompt = f"""
You are a helpful assistant for BigBasket users.

Always provide a clear, concise, and helpful response based on the context.
Do not say things like "We couldn’t find an exact match" or refer to missing information.
Avoid unnecessary disclaimers or comments about the data. Just give the best possible answer.

User question: "{user_query}"

Context:
{context}

Respond with only the helpful answer.
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()




def process_user_query(user_query):
    if is_structured_query(user_query):
        sql = generate_sql_with_gemini(user_query)
        result = run_sql_query(sql)
        if isinstance(result, str):
            context = ""  # Handle SQL error
        else:
            context = result.to_markdown(index=False) if not result.empty else ""
    else:
        matches = semantic_search(user_query)
        context = matches[0] if matches else ""

    return summarize_with_gemini(user_query, context)


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
