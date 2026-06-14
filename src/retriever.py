from embeddings import embed_texts

def retrieve(query, store):

    query_embedding = embed_texts([query])[0]

    return store.search(
        query_embedding,
        k=3
    )