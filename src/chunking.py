import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, chunk_size=250, overlap=80):
    tokens = encoder.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size

        chunk = encoder.decode(tokens[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks