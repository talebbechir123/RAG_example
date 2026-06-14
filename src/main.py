from ingest import load_documents
from chunking import chunk_text
from embeddings import embed_texts
from vector_store import VectorStore
from llm import ask_ollama

# 1. Load docs
docs = load_documents("data/raw")

# 2. Chunk
chunks = []
for doc in docs:
    chunks.extend(chunk_text(doc["text"]))

# 3. Embed + index
embeddings = embed_texts(chunks)

store = VectorStore(len(embeddings[0]))
store.add(chunks, embeddings)

print("RAG system ready!\n")

# 4. Chat loop
while True:
    question = input("\nQuestion: ")

    if question.lower() in ["exit", "quit"]:
        break

    # 5. Retrieve
    query_emb = embed_texts([question])[0]
    top_chunks = store.search(query_emb, k=3)

    context = "\n\n".join(top_chunks)

    # 6. Prompt
    prompt = f"""
You are a helpful assistant.
Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    # 7. Ask LLM
    answer = ask_ollama(prompt)

    print("\nAnswer:\n")
    print(answer)