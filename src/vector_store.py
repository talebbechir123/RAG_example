import faiss
import numpy as np

class VectorStore:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

        self.chunks = []

    def add(self, chunks, embeddings):

        self.index.add(
            np.array(embeddings).astype("float32")
        )

        self.chunks.extend(chunks)

    def search(self, embedding, k=3):

        distances, indices = self.index.search(
            np.array([embedding]).astype("float32"),
            k
        )

        return [self.chunks[i] for i in indices[0]]