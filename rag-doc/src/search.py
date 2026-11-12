
from pathlib import Path

import fitz
import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("data/sample.pdf")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    chunks = []

    with fitz.open(PDF_PATH) as document:
        for page_num, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            start = 0
            chunk_size = 800
            overlap = 120

            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()

                if chunk:
                    chunks.append({
                        "page": page_num,
                        "text": chunk
                    })

                if end == len(text):
                    break

                start = end - overlap

    return chunks


def main():
    chunks = load_chunks()

    if not chunks:
        print("No text chunks found. Check data/sample.pdf.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    vectors = np.asarray(vectors, dtype="float32")

    # Cosine similarity via inner product of normalized vectors
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print("\nSemantic search is ready.")
    question = input("Ask a question about the PDF: ").strip()

    if not question:
        print("No question entered.")
        return

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    k = min(3, len(chunks))
    scores, ids = index.search(query_vector, k)

    print("\nMost relevant chunks:")

    for rank, (score, idx) in enumerate(
        zip(scores[0], ids[0]), start=1
    ):
        item = chunks[int(idx)]

        print(f"\n--- Result {rank} ---")
        print(f"Page: {item['page']}")
        print(f"Similarity: {float(score):.3f}")
        print(item["text"][:500])


if __name__ == "__main__":
    main()
