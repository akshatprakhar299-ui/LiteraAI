
import json
from pathlib import Path

import faiss


INDEX_PATH = Path("data/indexes/faiss.index")
METADATA_PATH = Path("data/metadata/chunks.json")


def save_store(index, chunks):
    """Save FAISS index and corresponding chunks."""

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunks count do not match.")

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print("FAISS index and metadata saved successfully.")


def load_store():
    """Load previously saved FAISS index and chunks."""

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        print("No existing index found.")
        return None, None

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and metadata are inconsistent.")

    print(f"Loaded existing index with {index.ntotal} vectors.")

    return index, chunks
