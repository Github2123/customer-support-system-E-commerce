BigBasket AI Assistant
Welcome to the BigBasket AI Assistant project! This assistant uses state-of-the-art AI and machine learning technologies to help users find answers to their queries about BigBasket's products, prices, brands, and policies.

Technologies Used
Streamlit: For creating the web-based user interface (UI).

Sentence Transformers: For generating embeddings (vector representations) of user queries.

FAISS: For performing high-performance similarity searches on the FAQ data.

Gemini API (Google): For generating natural language responses and SQL queries.

SQLite: For managing and querying product data.

Pandas: For handling and displaying query results in a readable format.




How It Works
User Query Input: The user enters a query in the Streamlit interface.

Query Handling:

If the query is structured (e.g., asking for prices or top-rated products), the assistant generates an SQL query using the Gemini API.

If the query is unstructured (e.g., general questions), the assistant performs a semantic search on the FAQ data using FAISS and Sentence Transformers.

Response Generation:

The assistant uses the Gemini API to generate a response based on the query and the context from either the FAQ or database query results.


Features
Semantic Search: Uses Sentence Transformers and FAISS for efficient text similarity search on the FAQ data.

SQL Query Generation: Uses Google Gemini API to generate SQLite queries based on user input, for example, to retrieve product information.

Interactive UI: Built with Streamlit to allow easy user interaction.

Future Enhancements
Improve the semantic search results with more advanced embeddings.

Integrate more datasets for a wider variety of queries.

Add more context-aware responses using the Gemini API.

