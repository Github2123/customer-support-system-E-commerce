import pandas as pd
import sqlite3

# Connect to SQLite Database (or create it)
conn = sqlite3.connect('bigbasket.db')
cursor = conn.cursor()

# Create table with actual columns from your CSV
cursor.execute('''
CREATE TABLE IF NOT EXISTS bigbasket_data (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    sub_category TEXT,
    brand TEXT,
    sale_price REAL,
    market_price REAL,
    type TEXT,
    rating REAL,
    description TEXT
);
''')


df = pd.read_csv('knowledge_Base1/BigBasket.csv', encoding='ISO-8859-1')  # or encoding='cp1252'


for idx, row in df.iterrows():
    cursor.execute('''
        INSERT OR REPLACE INTO bigbasket_data 
        (product_id, product_name, category, sub_category, brand, sale_price, market_price, type, rating, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    ''', (
        int(row['index']),              # using CSV 'index' column as primary key
        row['product'],
        row['category'],
        row['sub_category'],
        row['brand'],
        float(row['sale_price']),
        float(row['market_price']),
        row['type'],
        float(row['rating']) if not pd.isnull(row['rating']) else None,
        row['description']
    ))

# Commit and close connection
conn.commit()
conn.close()

print("✅ Data stored successfully in the SQL database.")
