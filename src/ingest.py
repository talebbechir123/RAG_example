from pathlib import Path

def load_documents(data_dir):
    documents = []

    for file in Path(data_dir).glob("*.txt"):
        text = file.read_text(encoding="utf-8")

        documents.append({
            "source": file.name,
            "text": text
        })

    return documents