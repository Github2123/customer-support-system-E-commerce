import streamlit as st
import pandas as pd
import faiss
import numpy as np
import sqlite3
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import logging
import os
from datetime import datetime

# 🔐 Configure Logging Folder
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)

# 🔐 Configure Logging to a file within the logs folder
logging.basicConfig(
    filename=os.path.join(log_folder, "bigbasket_ai_assistant.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Logging configured.")

# 🔐 Gemini API Authentication
genai.configure(api_key="AIzaSyCqAZ_f8Lvn-i_jX2phis1ASD6rZWN25Q4")
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# 📚 Load data
logging.info("Loading model and FAISS index")
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index('bigbasket_faq_index.faiss')
with open('knowledge_Base1/BigBasketFAQ.md', 'r', encoding='utf-8') as f:
    faq_texts = f.read().split('\n\n')
logging.info("Data loaded successfully")

# ⚙️ Utility functions
def is_structured_query(query):
    keywords = ['top', 'lowest', 'highest', 'price', 'under', 'above', 'brand', 'category']
    return any(word in query.lower() for word in keywords)

def semantic_search(query, k=1):
    try:
        vector = model.encode([query]).astype('float32')
        D, I = faiss_index.search(vector, k)
        return [faq_texts[i] for i in I[0]]
    except Exception as e:
        logging.error(f"FAISS search error: {str(e)}")
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
    logging.info(f"Generating SQL for query: {user_query}")
    response = gemini_model.generate_content(prompt)
    raw_sql = response.text.strip()
    if raw_sql.startswith("```"):
        raw_sql = raw_sql.strip("`")
        raw_sql = raw_sql.replace("sql", "", 1).strip()
    logging.info(f"Generated SQL: {raw_sql}")
    return raw_sql

def run_sql_query(query):
    try:
        logging.info(f"Executing SQL query: {query}")
        conn = sqlite3.connect("bigbasket.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        logging.info("SQL query executed successfully")
        return df
    except Exception as e:
        logging.error(f"Error executing SQL: {e}")
        return f"Error executing SQL: {e}"

def summarize_with_gemini(user_query, context):
    prompt = f"""
You are a helpful assistant for BigBasket users.

If the retrieved data is not relevant to the question or seems incomplete, respond with a helpful message based on your general knowledge. You can also say things like "We couldn’t find an exact match, but here’s what might help..." or provide general tips related to the topic.

User asked: "{user_query}"

Here is the retrieved information:
{context}

Please provide a helpful and clear answer based on this.
"""
    logging.info(f"Summarizing context for query: {user_query}")
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def store_token_cost(model, prompt_tokens, completion_tokens):
    # Define cost for input tokens
    if prompt_tokens <= 128000:
        input_cost = prompt_tokens * 0.075 / 1000000  # $0.075 per million tokens
    else:
        input_cost = prompt_tokens * 0.15 / 1000000  # $0.15 per million tokens

    # Define cost for output tokens
    if completion_tokens <= 128000:
        output_cost = completion_tokens * 0.30 / 1000000  # $0.30 per million tokens
    else:
        output_cost = completion_tokens * 0.60 / 1000000  # $0.60 per million tokens

    # Total cost in USD
    total_cost = round(input_cost + output_cost, 6)

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(""" 
            INSERT INTO trip_costs (timestamp, model, prompt_tokens, completion_tokens, total_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            model,
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            total_cost
        ))
        conn.commit()
        conn.close()
        logging.info(f"Token cost stored: {prompt_tokens}+{completion_tokens} = {total_cost}")
    except Exception as e:
        logging.error(f"Failed to store token cost: {e}")

def process_user_query(user_query):
    logging.info(f"Processing user query: {user_query}")
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

    # Generate response from Gemini
    assistant_response = summarize_with_gemini(user_query, context)

    # Calculate tokens used
    prompt_tokens = len(user_query.split())  # Simple token count for the prompt
    completion_tokens = len(assistant_response.split())  # Simple token count for the completion

    # Store token cost after generating the response
    store_token_cost("gemini-1.5-pro", prompt_tokens, completion_tokens)

    return assistant_response

# 🎨 Streamlit App UI
st.set_page_config(page_title="BigBasket AI Assistant", page_icon="🛒")
st.title("🤖 BigBasket AI Assistant")
st.markdown("Ask me anything about products, brands, prices, or policies!")

query = st.text_input("💬 Your Question:", placeholder="e.g., Show me top 5 rated snacks under ₹100")

if query:
    logging.info(f"User queried: {query}")
    with st.spinner("Thinking..."):
        response = process_user_query(query)
        st.markdown("### 💡 Assistant Response:")
        st.markdown(response)
    logging.info(f"Response generated: {response}")
