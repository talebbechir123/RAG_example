# RAG Example — Local Retrieval-Augmented Generation in Python

A minimal, fully local Retrieval-Augmented Generation (RAG) pipeline: documents are chunked by tokens, embedded with Sentence-Transformers, indexed in FAISS, and the top matches are passed as context to a Llama 3 model served by Ollama. No external API is called.

The sample corpus is the Tiny Shakespeare dataset (`DATA/raw/input.txt`, 40,000 lines).

## Pipeline

```
 .txt files ──► token chunking ──► embeddings ──► FAISS index
                (250 tok, 80 overlap)  (MiniLM-L6-v2)   (exact L2)
                                                          │
 question ──► embed ──► top-3 chunks ──► prompt ──► Llama 3 (Ollama) ──► answer
```

| Module | Role |
|---|---|
| `src/ingest.py` | Loads every `.txt` file of a directory |
| `src/chunking.py` | Splits text into 250-token windows with 80-token overlap (`tiktoken`, `cl100k_base`) |
| `src/embeddings.py` | Encodes text with `sentence-transformers/all-MiniLM-L6-v2` |
| `src/vector_store.py` | FAISS `IndexFlatL2` wrapper keeping chunks aligned with vectors |
| `src/retriever.py` | Embeds a query and returns the k nearest chunks |
| `src/llm.py` | Calls the Ollama `/api/generate` endpoint |
| `src/build_index.py` | Builds the index and reports the number of chunks |
| `src/main.py` | Interactive question-answering loop |
| `src/eval_data.py` | Question / ground-truth pairs for evaluating answers |

The prompt instructs the model to answer **only** from the retrieved context, which limits hallucination and makes answers traceable to source chunks.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the `llama3` model

```bash
pip install sentence-transformers faiss-cpu tiktoken numpy requests
ollama pull llama3
```

## Usage

The scripts read documents from `data/raw`, relative to the working directory. On a case-sensitive file system (Linux), rename `DATA/` to `data/` first.

```bash
python src/build_index.py   # index the corpus and print the chunk count
python src/main.py          # ask questions; type "exit" to quit
```

To use your own documents, drop `.txt` files into `data/raw/`.

## Possible extensions

- Persist the FAISS index instead of rebuilding it at each start
- Score answers against `eval_data.py` (e.g. with RAGAS or embedding similarity)
- Swap `IndexFlatL2` for an approximate index (IVF, HNSW) on larger corpora

## Author

**Ahmed Taleb Bechir**
[atbechir.com](https://atbechir.com) · [LinkedIn](https://www.linkedin.com/in/ahmed-taleb-bechir-97a7301a4) · [GitHub](https://github.com/talebbechir123)
