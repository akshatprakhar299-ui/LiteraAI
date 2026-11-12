
import json
from pathlib import Path

import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from persistence import load_store
from evaluate import TEST_CASES
from generate import tokenize



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3
RRF_K = 60

REPORT_PATH = Path("data/reports/diagnostic_report.json")
EVALUATION_SOURCE = "sample.pdf"



def build_bm25(chunks):

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def retrieve_with_scores(question, chunks, model, index, bm25):

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    _, semantic_ids = index.search(
        query_vector,
        len(chunks)
    )

    keyword_scores = bm25.get_scores(
        tokenize(question)
    )

    keyword_ids = np.argsort(keyword_scores)[::-1]

    fused_scores = {}

    for ranking in (semantic_ids[0], keyword_ids):

        for rank, chunk_id in enumerate(ranking, start=1):

            chunk_id = int(chunk_id)

            if chunk_id < 0:
                continue

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (RRF_K + rank)
            )

    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    results = []

    for chunk_id in ranked_ids[:TOP_K]:

        results.append({
            "source": chunks[chunk_id]["source"],
            "page": chunks[chunk_id]["page"],
            "chunk": chunks[chunk_id].get("chunk"),
            "rrf_score": round(
                fused_scores[chunk_id], 6
            ),
            "text": chunks[chunk_id]["text"]
        })

    return results



def diagnose_case(case, results):

    question = case["question"]

    expected_pages = set(case["expected_pages"])

    retrieved_pages = {
        result["page"]
        for result in results
        if result["source"] == EVALUATION_SOURCE
    }

    if not expected_pages:

        return {
            "question": question,
            "type": "unanswerable",
            "retrieved_pages": sorted(retrieved_pages),
            "status": "not_evaluated",
            "diagnosis": (
                "Retrieval metrics cannot determine whether "
                "the generated answer correctly abstains. "
                "Answer-level abstention testing is required."
            )
        }

    found_pages = expected_pages & retrieved_pages
    missed_pages = expected_pages - retrieved_pages

    recall = len(found_pages) / len(expected_pages)

    hit = len(found_pages) > 0

    if recall == 1:
        status = "excellent"
        diagnosis = "All expected evidence pages were retrieved."

    elif hit:
        status = "partial"
        diagnosis = "Some expected evidence pages were missed."

    else:
        status = "failed"
        diagnosis = "No expected evidence pages were retrieved."

    return {
        "question": question,
        "type": "answerable",
        "expected_pages": sorted(expected_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "found_pages": sorted(found_pages),
        "missed_pages": sorted(missed_pages),
        "hit": hit,
        "recall": round(recall, 4),
        "status": status,
        "diagnosis": diagnosis,
        "retrieved_chunks": [
            {
                "source": result["source"],
                "page": result["page"],
                "chunk": result["chunk"],
                "rrf_score": result["rrf_score"]
            }
            for result in results
        ]
    }



def generate_recommendations(results):

    answerable = [
        result for result in results
        if result["type"] == "answerable"
    ]

    recommendations = []

    if not answerable:
        return ["Add answerable evaluation questions."]

    failed = [
        result for result in answerable
        if result["status"] == "failed"
    ]

    partial = [
        result for result in answerable
        if result["status"] == "partial"
    ]

    if failed:
        recommendations.append(
            "Investigate completely missed evidence. "
            "Consider improving chunk boundaries, query "
            "formulation and retrieval ranking."
        )

    if partial:
        recommendations.append(
            "Some relevant pages are missing. Consider "
            "increasing retrieval depth and testing "
            "different chunk sizes."
        )

    if not failed and not partial:
        recommendations.append(
            "All current answerable test cases retrieved "
            "their expected evidence pages. Test on a larger "
            "and more diverse evaluation dataset."
        )

    if len(answerable) < 20:
        recommendations.append(
            "Expand the evaluation dataset. A small number "
            "of test questions cannot reliably establish "
            "general retrieval quality."
        )

    recommendations.append(
        "Tune retrieval parameters only after comparing "
        "results on a separate validation dataset."
    )

    return recommendations


def run_diagnostics():

    print("=" * 60)
    print("             RAG DOCTOR")
    print("          DIAGNOSTIC ENGINE")
    print("=" * 60)

    # Load saved index

    print("\nLoading persistent storage...")

    index, chunks = load_store()

    if index is None or chunks is None:
        raise RuntimeError(
            "No saved index found. Run ingestion first."
        )

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS and metadata counts do not match."
        )

    # Load embedding model

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # Build BM25

    print("\nBuilding BM25 retriever...")

    bm25 = build_bm25(chunks)

    # Run tests

    print("\nRunning diagnostic tests...")

    results = []

    for number, case in enumerate(TEST_CASES, start=1):

        print(f"\nTest {number}: {case['question']}")

        retrieved = retrieve_with_scores(
            case["question"],
            chunks,
            model,
            index,
            bm25
        )

        diagnosis = diagnose_case(case, retrieved)

        results.append(diagnosis)

        print(f"Status: {diagnosis['status']}")

        if diagnosis["type"] == "answerable":

            print(
                f"Recall@{TOP_K}: "
                f"{diagnosis['recall']:.1%}"
            )

            print(
                f"Missed pages: "
                f"{diagnosis['missed_pages']}"
            )

    # Calculate metrics

    answerable = [
        result for result in results
        if result["type"] == "answerable"
    ]

    evaluated = len(answerable)

    hit_rate = (
        sum(result["hit"] for result in answerable) / evaluated
        if evaluated else 0
    )

    mean_recall = (
        sum(result["recall"] for result in answerable) / evaluated
        if evaluated else 0
    )

    # Generate recommendations

    recommendations = generate_recommendations(results)

    report = {
        "project": "RAG Doctor",
        "evaluation_source": EVALUATION_SOURCE,
        "retrieval_method": "FAISS + BM25 + RRF",
        "top_k": TOP_K,
        "total_test_cases": len(results),
        "answerable_cases": evaluated,
        "unanswerable_cases": len(results) - evaluated,
        "metrics": {
            "hit_rate_at_3": round(hit_rate, 4),
            "mean_evidence_page_recall_at_3": round(
                mean_recall, 4
            )
        },
        "diagnostic_results": results,
        "recommendations": recommendations,
        "limitations": [
            "Metrics use page-level relevance labels.",
            "Unanswerable questions are not yet tested "
            "for answer-level abstention.",
            "Citation correctness and factual support "
            "are not evaluated by this module."
        ]
    }

    # Save report

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Display final report

    print("\n" + "=" * 60)
    print("          DIAGNOSTIC SUMMARY")
    print("=" * 60)

    print(f"Answerable tests: {evaluated}")

    print(f"Hit Rate@{TOP_K}: {hit_rate:.1%}")

    print(
        f"Mean Evidence Recall@{TOP_K}: "
        f"{mean_recall:.1%}"
    )

    print("\nRECOMMENDATIONS:")

    for number, recommendation in enumerate(
        recommendations, start=1
    ):

        print(f"{number}. {recommendation}")

    print(f"\nFull report saved to: {REPORT_PATH}")

    print("\nDiagnostic evaluation completed.")


if __name__ == "__main__":
    run_diagnostics()
