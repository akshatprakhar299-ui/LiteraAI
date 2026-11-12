"""
LiteraAI unified RAG pipeline.

Reuses the existing retrieval, generation, citation, evaluation,
and ingestion logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from persistence import load_store, save_store
from answer_evaluator import evaluate_answer, correct_answer
from generate import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    build_keyword_retriever,
    generate_answer,
    retrieve,
    verify_citations,
)
from research_search import search_research_papers
from ingest import calculate_hash, process_document, create_embeddings


class RagPipeline:
    """Load resources once and serve multiple requests."""

    def __init__(self) -> None:
        self.semantic_index, self.chunks = load_store()

        if self.semantic_index is None or not self.chunks:
            raise RuntimeError(
                "No saved index found. Check data/indexes and data/metadata."
            )

        if self.semantic_index.ntotal != len(self.chunks):
            raise RuntimeError("FAISS and metadata counts do not match.")

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.bm25 = build_keyword_retriever(self.chunks)

    # ---------------------------------------------------------
    # MULTI-PDF INGESTION
    # ---------------------------------------------------------

    def ingest_pdfs(
        self,
        uploaded_files: List[Tuple[str, bytes]],
    ) -> Dict[str, Any]:
        if not uploaded_files:
            raise ValueError("No PDF files were provided.")

        known_hashes = {
            chunk["document_hash"]
            for chunk in self.chunks
            if chunk.get("document_hash")
        }
        known_sources = {
            chunk["source"]
            for chunk in self.chunks
            if chunk.get("source")
        }

        new_chunks = []
        processed_documents = []
        skipped_documents = []

        with tempfile.TemporaryDirectory(prefix="literaai_upload_") as temp_directory:
            root = Path(temp_directory)

            for number, (filename, content) in enumerate(uploaded_files, start=1):
                safe_name = Path(filename).name

                if not safe_name.lower().endswith(".pdf"):
                    raise ValueError(f"{safe_name} is not a PDF file.")
                if not content:
                    raise ValueError(f"{safe_name} is empty.")

                document_directory = root / str(number)
                document_directory.mkdir(parents=True)
                pdf_path = document_directory / safe_name
                pdf_path.write_bytes(content)

                document_hash = calculate_hash(pdf_path)

                if document_hash in known_hashes:
                    skipped_documents.append({
                        "filename": safe_name,
                        "reason": "Duplicate file content",
                    })
                    continue

                if safe_name in known_sources:
                    skipped_documents.append({
                        "filename": safe_name,
                        "reason": "A document with this filename already exists in the index",
                    })
                    continue

                document_chunks = process_document(pdf_path)
                if not document_chunks:
                    skipped_documents.append({
                        "filename": safe_name,
                        "reason": "No extractable text found",
                    })
                    continue

                new_chunks.extend(document_chunks)
                known_hashes.add(document_hash)
                known_sources.add(safe_name)
                processed_documents.append({
                    "filename": safe_name,
                    "document_hash": document_hash,
                    "chunks": len(document_chunks),
                })

        if not new_chunks:
            return {
                "status": "no_new_documents",
                "message": "No new documents were added.",
                "processed_documents": [],
                "skipped_documents": skipped_documents,
                "new_chunks": 0,
                "total_chunks": len(self.chunks),
            }

        vectors = create_embeddings(new_chunks, self.model)
        updated_index = faiss.clone_index(self.semantic_index)

        if updated_index.d != vectors.shape[1]:
            raise ValueError(
                "Embedding dimensions do not match the existing FAISS index."
            )

        updated_index.add(vectors)
        updated_chunks = self.chunks + new_chunks
        save_store(updated_index, updated_chunks)

        self.semantic_index = updated_index
        self.chunks = updated_chunks
        self.bm25 = build_keyword_retriever(self.chunks)

        return {
            "status": "success",
            "message": "PDF ingestion completed successfully.",
            "processed_documents": processed_documents,
            "skipped_documents": skipped_documents,
            "new_chunks": len(new_chunks),
            "total_chunks": len(updated_chunks),
            "total_vectors": self.semantic_index.ntotal,
        }

    # ---------------------------------------------------------
    # DOCUMENT QUESTION ANSWERING + CLAIM VERIFICATION
    # ---------------------------------------------------------

    def answer(
        self,
        question: str,
        evaluate: bool = True,
    ) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        results = retrieve(
            question,
            self.chunks,
            self.model,
            self.semantic_index,
            self.bm25,
        )

        if not results:
            return {
                "question": question,
                "answer": "I could not find relevant evidence in your documents.",
                "model": LLM_MODEL,
                "sources": [],
                "citation_check": None,
                "evaluation": None,
                "answer_revised": False,
            }

        answer_text = generate_answer(question, results)
        citation_check = verify_citations(answer_text, results)

        evaluation = None
        initial_evaluation = None
        answer_revised = False
        correction_error = None

        if evaluate:
            initial_evaluation = evaluate_answer(
                question,
                answer_text,
                results,
            )
            evaluation = initial_evaluation

            needs_correction = (
                initial_evaluation["unsupported"] > 0
                or initial_evaluation["contradicted"] > 0
                or initial_evaluation["uncertain"] > 0
            )

            if needs_correction:
                try:
                    revised_answer = correct_answer(
                        question,
                        answer_text,
                        results,
                        initial_evaluation,
                    )

                    # Only accept the revised answer if it can be evaluated.
                    revised_evaluation = evaluate_answer(
                        question,
                        revised_answer,
                        results,
                    )

                    answer_text = revised_answer
                    evaluation = revised_evaluation
                    answer_revised = True
                    citation_check = verify_citations(answer_text, results)

                except (ValueError, RuntimeError):
                    # Preserve the original answer and its evaluation if
                    # correction fails. Never hide the failed correction.
                    correction_error = (
                        "Automatic correction could not be completed. "
                        "Review the unsupported or uncertain claims."
                    )

        sources = [
            {
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "chunk": chunk.get("chunk"),
                "text": chunk.get("text"),
            }
            for chunk in results
        ]

        return {
            "question": question,
            "answer": answer_text,
            "model": LLM_MODEL,
            "sources": sources,
            "citation_check": citation_check,
            "evaluation": evaluation,
            "initial_evaluation": initial_evaluation,
            "answer_revised": answer_revised,
            "correction_error": correction_error,
        }

    # ---------------------------------------------------------
    # RESEARCH PAPER DISCOVERY
    # ---------------------------------------------------------

    def find_papers(
        self,
        query: str,
        document_context: Optional[str] = None,
        **search_kwargs: Any,
    ) -> Dict[str, Any]:
        if document_context is None:
            top_chunks = retrieve(
                query,
                self.chunks,
                self.model,
                self.semantic_index,
                self.bm25,
            )
            document_context = " ".join(
                chunk["text"] for chunk in top_chunks
            )

        return search_research_papers(
            user_query=query,
            document_context=document_context,
            **search_kwargs,
        )

    # ---------------------------------------------------------
    # ANSWER + RELATED PAPERS
    # ---------------------------------------------------------

    def answer_with_papers(
        self,
        question: str,
        paper_limit: int = 5,
    ) -> Dict[str, Any]:
        result = self.answer(question)
        context = " ".join(
            source["text"] for source in result["sources"]
        )

        try:
            papers = search_research_papers(
                user_query=question,
                document_context=context,
                limit=paper_limit,
            )
        except (ValueError, RuntimeError) as error:
            papers = {"error": str(error), "papers": []}

        result["related_papers"] = papers
        return result
