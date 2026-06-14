from ingest import load_documents
from chunking import chunk_text
from embeddings import embed_texts
from vector_store import VectorStore

docs = load_documents("data/raw")

all_chunks = []

for doc in docs:

    chunks = chunk_text(doc["text"])

    all_chunks.extend(chunks)

embeddings = embed_texts(all_chunks)

dimension = len(embeddings[0])

store = VectorStore(dimension)

store.add(all_chunks, embeddings)

print(
    f"Indexed {len(all_chunks)} chunks"
)