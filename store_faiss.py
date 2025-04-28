from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAQs
with open('knowledge_Base1/BigBasketFAQ.md', 'r') as file:
    faq_data = file.read().split('\n\n')

# Generate embeddings
faq_embeddings = [model.encode(faq) for faq in faq_data]
faq_embeddings = np.array(faq_embeddings).astype('float32')

# Get actual dimension from sample
dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add to FAISS index
index.add(faq_embeddings)
faiss.write_index(index, 'bigbasket_faq_index.faiss')

print("✅ FAQs data stored successfully in FAISS vector database.")
