from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.helper import load_pdf, text_splitter

load_dotenv()
# manual index creation
# Load your API key and environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical"

# Load and split documents
documents = load_pdf("data")
texts = text_splitter(documents)
print(f"✅ Loaded and split {len(texts)} chunks.")

# Embed texts using HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedded_texts = embedding_model.embed_documents([t.page_content for t in texts])

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Create index if it doesn't exist
if INDEX_NAME not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=len(embedded_texts[0]), metric='cosine', spec=spec)
    print(f"✅ Index '{INDEX_NAME}' created.")
else:
    print(f"ℹ️ Index '{INDEX_NAME}' already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# Prepare vectors for upsert
vectors = []
for i, (text, embedding) in enumerate(zip(texts, embedded_texts)):
    vectors.append({
        "id": f"doc-{i}",
        "values": embedding,
        "metadata": {
            "text": text.page_content
        }
    })

# Upsert to Pinecone
index.upsert(vectors)
print(f"✅ Upserted {len(vectors)} vectors to Pinecone index '{INDEX_NAME}'.")
