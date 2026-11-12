
import re
import requests
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from persistence import load_store
from answer_evaluator import evaluate_answer, display_evaluation

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"

OLLAMA_URL = "http://localhost:11434/api/chat"

TOP_K = 3
RRF_K = 60



def tokenize(text):
    """Convert text into lowercase word tokens."""

    return re.findall(r"\b\w+\b", text.lower())


def build_keyword_retriever(chunks):
    """Build BM25 keyword retriever from saved chunks."""

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)

def reciprocal_rank_fusion(semantic_ids, keyword_ids):
    """Combine semantic and keyword rankings using RRF."""

    fused_scores = {}

    for ranking in (semantic_ids, keyword_ids):

        for rank, chunk_id in enumerate(ranking, start=1):

            chunk_id = int(chunk_id)

            if chunk_id < 0:
                continue

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    return ranked_ids


def retrieve(question, chunks, model, semantic_index, bm25):
    """Retrieve relevant chunks using hybrid search."""

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    _, semantic_ids = semantic_index.search(
        query_vector,
        len(chunks)
    )

    keyword_scores = bm25.get_scores(
        tokenize(question)
    )

    keyword_ids = np.argsort(keyword_scores)[::-1]

    ranked_ids = reciprocal_rank_fusion(
        semantic_ids[0],
        keyword_ids
    )

    return [
        chunks[chunk_id]
        for chunk_id in ranked_ids[:TOP_K]
    ]


def generate_answer(question, retrieved_chunks):
    """Generate a grounded answer using local Llama."""

    context = "\n\n".join(
        f"[Source: {chunk['source']}, Page {chunk['page']}]\n"
        f"{chunk['text']}"
        for chunk in retrieved_chunks
    )

    system_prompt = """
You are RAG Doctor, a document question-answering assistant.

RULES:

1. Answer using only the supplied PDF context.
2. Never use outside knowledge.
3. If evidence is insufficient, explicitly say:
   "I couldn't find enough information in the provided PDFs to answer that."
4. Cite evidence using:
   [Source: filename.pdf, Page N]
5. Never invent facts, quotations or page numbers.
6. Keep answers clear and direct.
7. Use multiple documents when relevant.
8. Distinguish facts from interpretations.
"""

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED PDF CONTEXT:
{context}

Answer using only the supplied evidence.
Include appropriate source and page citations.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()



def verify_citations(answer, retrieved_chunks):
    """Check whether cited filenames and pages were retrieved."""

    retrieved_sources = {
        (chunk["source"], chunk["page"])
        for chunk in retrieved_chunks
    }

    citations = re.findall(
        r"\[Source:\s*(.*?),\s*Page\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE
    )

    cited_sources = {
        (source.strip(), int(page))
        for source, page in citations
    }

    if not cited_sources:
        return {
            "valid": False,
            "message": "No document citations found.",
            "cited_sources": [],
            "invalid_sources": []
        }

    invalid_sources = cited_sources - retrieved_sources

    if invalid_sources:
        return {
            "valid": False,
            "message": "Some citations do not match retrieved evidence.",
            "cited_sources": sorted(cited_sources),
            "invalid_sources": sorted(invalid_sources)
        }

    return {
        "valid": True,
        "message": "All citations match retrieved document pages.",
        "cited_sources": sorted(cited_sources),
        "invalid_sources": []
    }



def display_citations(verification):

    print("\n" + "=" * 60)
    print("CITATION VERIFICATION")
    print("=" * 60)

    print(verification["message"])

    print(f"Cited sources: {verification['cited_sources']}")

    if verification["invalid_sources"]:
        print(
            f"Invalid citations: "
            f"{verification['invalid_sources']}"
        )



def main():

    print("=" * 60)
    print("                    RAG DOCTOR")
    print("=" * 60)

    # STEP 1: Load persistent storage

    print("\nLoading saved FAISS index and chunks...")

    semantic_index, chunks = load_store()

    if semantic_index is None or chunks is None:
        print("No saved index found.")
        print("Run: python src/ingest.py")
        return

    if not chunks:
        print("No document chunks found.")
        return

    if semantic_index.ntotal != len(chunks):
        print("FAISS and metadata counts do not match.")
        return

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loaded vectors: {semantic_index.ntotal}")

    documents = sorted({
        chunk["source"] for chunk in chunks
    })

    print(f"Indexed documents: {len(documents)}")

    for document in documents:
        print(f"  - {document}")

    # STEP 2: Load embedding model

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model loaded.")

    # STEP 3: Build BM25

    print("\nBuilding BM25 keyword retriever...")

    bm25 = build_keyword_retriever(chunks)

    print("BM25 retriever ready.")

    # STEP 4: Start application

    print("\n" + "=" * 60)
    print("RAG DOCTOR IS READY")
    print(f"LLM: {LLM_MODEL}")
    print("Retrieval: FAISS + BM25 + RRF")
    print("Evaluation: Citation + Claim-level evidence")
    print("Type 'exit' to quit.")
    print("=" * 60)

    # STEP 5: Question-answering loop

    while True:

        question = input(
            "\nAsk a question about your PDFs: "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not question:
            continue

        try:

            # RETRIEVAL

            print("\nRetrieving relevant evidence...")

            results = retrieve(
                question,
                chunks,
                model,
                semantic_index,
                bm25
            )

            if not results:
                print("No evidence retrieved.")
                continue

            print("\nRETRIEVED EVIDENCE:")

            for rank, item in enumerate(results, start=1):

                print(
                    f"{rank}. {item['source']} | "
                    f"Page {item['page']} | "
                    f"Chunk {item['chunk']}"
                )

            # ANSWER GENERATION

            print("\nGenerating answer using Llama...")

            answer = generate_answer(
                question,
                results
            )

            print("\n" + "=" * 60)
            print("RAG DOCTOR'S ANSWER")
            print("=" * 60)

            print(answer)

            # CITATION VERIFICATION

            verification = verify_citations(
                answer,
                results
            )

            display_citations(verification)

            # CLAIM-LEVEL EVALUATION

            print("\n" + "=" * 60)
            print("RUNNING HALLUCINATION EVALUATION")
            print("=" * 60)

            evaluation = evaluate_answer(
                question,
                answer,
                results
            )

            display_evaluation(evaluation)

        except requests.RequestException as error:

            print(f"\nOllama connection error: {error}")
            print("Check that Ollama is running.")

        except Exception as error:

            print(f"\nApplication error: {error}")



if __name__ == "__main__":
    main()
