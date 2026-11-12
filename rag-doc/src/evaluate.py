
from pathlib import Path
import re
import fitz
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("data/sample.pdf")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

TEST_CASES = [
    {
        "question": "What are the four steps of the improvement process?",
        "expected_pages": [4, 5],
    },
    {
        "question": "What are the six stages of the repeating pattern?",
        "expected_pages": [4],
    },
    {
        "question": "How does feedback help improve ChatGPT?",
        "expected_pages": [2, 9],
    },
    {
        "question": "What is the purpose of testing an improvement?",
        "expected_pages": [4, 5],
    },
    {
        "question": "What exact internal software does OpenAI use to manage its Agile sprints?",
        "expected_pages": [],
    },
]


def load_chunks():
    chunks = []

    with fitz.open(PDF_PATH) as document:
        for page_num, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            chunk_size = 800
            overlap = 120
            start = 0

            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()

                if chunk:
                    chunks.append({
                        "page": page_num,
                        "text": chunk,
                    })

                if end == len(text):
                    break

                start = end - overlap

    return chunks


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def build_semantic_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def reciprocal_rank_fusion(semantic_ids, keyword_ids, k=60):
    """
    Combine rankings without comparing incompatible raw scores.
    A result ranked highly by both methods receives a stronger score.
    """
    fused_scores = {}

    for ranking in (semantic_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            fused_scores[int(chunk_id)] = (
                fused_scores.get(int(chunk_id), 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )


def evaluate():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH.resolve()}")
        return

    print("Loading PDF chunks...")
    chunks = load_chunks()

    if not chunks:
        print("No text was extracted from the PDF.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building semantic and BM25 indexes...")
    semantic_index = build_semantic_index(chunks, model)

    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    total_hits = 0
    total_recall = 0.0
    evaluated_cases = 0

    print("\n" + "=" * 60)
    print(f"RAG DOCTOR — HYBRID RETRIEVAL EVALUATION (Recall@{TOP_K})")
    print("=" * 60)

    for case in TEST_CASES:
        question = case["question"]
        expected_pages = set(case["expected_pages"])

        query_vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        semantic_scores, semantic_ids = semantic_index.search(
            query_vector, len(chunks)
        )

        keyword_scores = bm25.get_scores(tokenize(question))
        keyword_ids = np.argsort(keyword_scores)[::-1]

        fused_ids = reciprocal_rank_fusion(
            semantic_ids[0],
            keyword_ids,
        )[:TOP_K]

        results = [
            {
                "page": chunks[chunk_id]["page"],
                "score": (
                    1.0 / (60 + list(fused_ids).index(chunk_id) + 1)
                ),
                "text": chunks[chunk_id]["text"],
            }
            for chunk_id in fused_ids
        ]

        retrieved_pages = {result["page"] for result in results}

        print(f"\nQuestion: {question}")
        print(f"Expected evidence pages: {sorted(expected_pages)}")
        print(f"Retrieved pages: {[r['page'] for r in results]}")

        if not expected_pages:
            print("Type: UNANSWERABLE — abstention not evaluated yet")
            continue

        evaluated_cases += 1
        found_pages = expected_pages & retrieved_pages
        recall = len(found_pages) / len(expected_pages)

        if found_pages:
            total_hits += 1

        total_recall += recall

        print(f"Missed evidence pages: {sorted(expected_pages - retrieved_pages)}")
        print(f"Evidence found: {'YES' if found_pages else 'NO'}")
        print(f"Recall@{TOP_K}: {recall:.2f}")

        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. Page {result['page']}")

    print("\n" + "=" * 60)
    print("HYBRID SEARCH SUMMARY")
    print("=" * 60)

    if evaluated_cases:
        print(f"Answerable questions evaluated: {evaluated_cases}")
        print(f"Hit rate@{TOP_K}: {total_hits / evaluated_cases:.1%}")
        print(f"Mean evidence-page Recall@{TOP_K}: {total_recall / evaluated_cases:.1%}")

    print("\nThese metrics measure retrieval, not answer correctness.")
    print("Compare the results with the previous 75% baseline.")


if __name__ == "__main__":
    evaluate()
