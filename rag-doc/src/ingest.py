
from pathlib import Path
import hashlib

import pymupdf
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from persistence import load_store, save_store

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def calculate_hash(pdf_path):
    """Generate SHA-256 hash for duplicate detection."""

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()

def extract_text(pdf_path):
    """Extract text page by page."""

    pages = []

    with pymupdf.open(pdf_path) as document:

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages

def split_into_chunks(text, chunk_size=800, overlap=120):

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks

def get_pdf_paths():

    print("\nEnter PDF paths one by one.")
    print("Type 'done' when finished.\n")

    pdf_paths = []
    seen_paths = set()

    while True:

        pdf_input = input("Enter PDF path: ").strip()

        if pdf_input.lower() == "done":
            break

        if not pdf_input:
            continue

        pdf_input = pdf_input.strip("\"'")
        pdf_path = Path(pdf_input).expanduser()

        if not pdf_path.exists() or not pdf_path.is_file():
            print("File not found. Try again.")
            continue

        if pdf_path.suffix.lower() != ".pdf":
            print("Please provide a PDF file.")
            continue

        pdf_path = pdf_path.resolve()

        if pdf_path in seen_paths:
            print("This PDF has already been selected.")
            continue

        seen_paths.add(pdf_path)
        pdf_paths.append(pdf_path)

        print(f"Added: {pdf_path.name}")

    return pdf_paths


def process_document(pdf_path):

    print(f"\nProcessing: {pdf_path.name}")

    document_hash = calculate_hash(pdf_path)
    pages = extract_text(pdf_path)

    document_chunks = []

    for page in pages:

        page_chunks = split_into_chunks(page["text"])

        for chunk_number, chunk_text in enumerate(
            page_chunks, start=1
        ):

            document_chunks.append({
                "source": pdf_path.name,
                "document_hash": document_hash,
                "page": page["page"],
                "chunk": chunk_number,
                "text": chunk_text
            })

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(document_chunks)}")

    return document_chunks

def create_embeddings(chunks, model):

    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return np.asarray(vectors, dtype="float32")

def update_index(existing_index, new_vectors):

    if existing_index is None:

        index = faiss.IndexFlatIP(new_vectors.shape[1])
        index.add(new_vectors)

        return index

    if existing_index.d != new_vectors.shape[1]:
        raise ValueError("Embedding dimensions do not match.")

    existing_index.add(new_vectors)

    return existing_index


def main():

    print("=" * 60)
    print("             RAG DOCTOR")
    print("       MULTI-DOCUMENT INGESTION")
    print("=" * 60)

    # Step 1: Get multiple PDF paths

    pdf_paths = get_pdf_paths()

    if not pdf_paths:
        print("\nNo PDFs selected. Exiting.")
        return

    print(f"\nTotal PDFs selected: {len(pdf_paths)}")

    # Step 2: Load existing storage

    print("\nLoading existing storage...")

    existing_index, existing_chunks = load_store()

    if existing_index is None:
        existing_chunks = []

    print(f"Existing chunks: {len(existing_chunks)}")

    # Step 3: Detect previously indexed documents

    known_hashes = {
        chunk["document_hash"]
        for chunk in existing_chunks
        if chunk.get("document_hash")
    }

    # Backward compatibility with older metadata
    known_sources = {
        chunk["source"]
        for chunk in existing_chunks
    }

    new_chunks = []

    for pdf_path in pdf_paths:

        document_hash = calculate_hash(pdf_path)

        if document_hash in known_hashes:
            print(f"\nAlready indexed: {pdf_path.name}")
            continue

        if pdf_path.name in known_sources:
            print(f"\nSkipping existing legacy document: {pdf_path.name}")
            continue

        document_chunks = process_document(pdf_path)

        if not document_chunks:
            print(f"No readable text in {pdf_path.name}")
            continue

        new_chunks.extend(document_chunks)

        known_hashes.add(document_hash)
        known_sources.add(pdf_path.name)

    # Step 4: Check new documents

    if not new_chunks:

        print("\nNo new documents to index.")
        return

    print("\n" + "=" * 60)
    print(f"New chunks to embed: {len(new_chunks)}")
    print("=" * 60)

    # Step 5: Generate embeddings

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("\nGenerating embeddings...")

    new_vectors = create_embeddings(new_chunks, model)

    # Step 6: Update FAISS

    print("\nUpdating FAISS index...")

    updated_index = update_index(
        existing_index,
        new_vectors
    )

    # Step 7: Combine old and new metadata

    updated_chunks = existing_chunks + new_chunks

    # Step 8: Save everything

    print("\nSaving updated storage...")

    save_store(
        updated_index,
        updated_chunks
    )

    # Step 9: Summary

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)

    print(f"New chunks added: {len(new_chunks)}")
    print(f"Total chunks: {len(updated_chunks)}")
    print(f"Total FAISS vectors: {updated_index.ntotal}")


if __name__ == "__main__":
    main()
