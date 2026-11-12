
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import requests

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import RagPipeline
from generate import retrieve, generate_answer, verify_citations
from answer_evaluator import evaluate_answer
from research_search import search_research_papers


state: dict = {}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = RagPipeline()
    yield
    state.clear()


app = FastAPI(
    title="LiteraAI API",
    description="API for the LiteraAI research and document intelligence platform.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    evaluate: bool = True
    include_papers: bool = False


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1)


class PapersRequest(BaseModel):
    query: str = Field(..., min_length=1)
    document_context: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    open_access_only: bool = False
    sort_by: str = "relevance"


class CitationRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class AnswerEvaluationRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)

def get_pipeline() -> RagPipeline:
    pipeline = state.get("pipeline")

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized.",
        )

    return pipeline


def get_retrieved_chunks(pipeline: RagPipeline, question: str):
    return retrieve(
        question,
        pipeline.chunks,
        pipeline.model,
        pipeline.semantic_index,
        pipeline.bm25,
    )


def handle_pipeline_error(error: Exception):
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error))

    if isinstance(error, requests.RequestException):
        raise HTTPException(
            status_code=502,
            detail=f"LLM connection error: {error}",
        )

    if isinstance(error, RuntimeError):
        raise HTTPException(status_code=500, detail=str(error))

    raise HTTPException(status_code=500, detail=str(error))

@app.get("/")
def root():
    return {
        "name": "LiteraAI API",
        "status": "running",
        "version": "2.0.0",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    pipeline = get_pipeline()

    return {
        "status": "ok",
        "service": "LiteraAI",
        "chunks": len(pipeline.chunks),
        "vectors": pipeline.semantic_index.ntotal,
    }


@app.get("/statistics")
def statistics():
    pipeline = get_pipeline()

    documents = sorted({
        chunk.get("source", "Unknown")
        for chunk in pipeline.chunks
    })

    return {
        "total_chunks": len(pipeline.chunks),
        "total_vectors": pipeline.semantic_index.ntotal,
        "total_documents": len(documents),
        "documents": documents,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "llama3.1:8b",
    }

@app.post("/ingest")
async def ingest_documents(
    files: list[UploadFile] = File(...)
):
    pipeline = get_pipeline()

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one PDF.",
        )

    uploaded_files = []

    for file in files:
        filename = Path(file.filename or "").name

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"{filename} is not a PDF file.",
            )

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"{filename} is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the 20 MB limit.",
            )

        uploaded_files.append((filename, content))

    try:
        result = await run_in_threadpool(
            pipeline.ingest_pdfs,
            uploaded_files,
        )

        return {
            "status": "success",
            "message": "Document ingestion completed.",
            "result": result,
            "total_chunks": len(pipeline.chunks),
            "total_vectors": pipeline.semantic_index.ntotal,
        }

    except Exception as error:
        handle_pipeline_error(error)

@app.get("/index/status")
def index_status():
    pipeline = get_pipeline()

    return {
        "loaded": pipeline.semantic_index is not None,
        "total_vectors": pipeline.semantic_index.ntotal,
        "total_chunks": len(pipeline.chunks),
        "documents": sorted({
            chunk.get("source", "Unknown")
            for chunk in pipeline.chunks
        }),
    }


@app.post("/index/reload")
async def reload_index():
    try:
        # Create a new pipeline first. Keep the existing one
        # if loading the saved index fails.
        new_pipeline = await run_in_threadpool(RagPipeline)

        state["pipeline"] = new_pipeline

        return {
            "status": "success",
            "message": "Index reloaded successfully.",
            "total_chunks": len(new_pipeline.chunks),
            "total_vectors": new_pipeline.semantic_index.ntotal,
        }

    except Exception as error:
        handle_pipeline_error(error)

@app.post("/search")
async def search_documents(req: SearchRequest):
    pipeline = get_pipeline()
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        results = await run_in_threadpool(
            get_retrieved_chunks,
            pipeline,
            question,
        )

        return {
            "question": question,
            "total_results": len(results),
            "results": [
                {
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "chunk": chunk.get("chunk"),
                    "text": chunk.get("text"),
                }
                for chunk in results
            ],
        }

    except Exception as error:
        handle_pipeline_error(error)

@app.post("/generate")
async def generate(req: GenerateRequest):
    pipeline = get_pipeline()
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        results = await run_in_threadpool(
            get_retrieved_chunks,
            pipeline,
            question,
        )

        if not results:
            return {
                "question": question,
                "answer": "No evidence retrieved.",
                "sources": [],
                "citation_check": None,
            }

        answer = await run_in_threadpool(
            generate_answer,
            question,
            results,
        )

        citation_check = verify_citations(answer, results)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "chunk": chunk.get("chunk"),
                    "text": chunk.get("text"),
                }
                for chunk in results
            ],
            "citation_check": citation_check,
        }

    except Exception as error:
        handle_pipeline_error(error)

@app.post("/ask")
async def ask(req: AskRequest):
    pipeline = get_pipeline()
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        if req.include_papers:
            result = await run_in_threadpool(
                pipeline.answer_with_papers,
                question,
            )
        else:
            result = await run_in_threadpool(
                pipeline.answer,
                question,
                req.evaluate,
            )

        return result

    except Exception as error:
        handle_pipeline_error(error)


@app.post("/ask-with-papers")
async def ask_with_papers(req: AskRequest):
    pipeline = get_pipeline()
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        return await run_in_threadpool(
            pipeline.answer_with_papers,
            question,
        )

    except Exception as error:
        handle_pipeline_error(error)

@app.post("/citations/verify")
async def verify_answer_citations(req: CitationRequest):
    pipeline = get_pipeline()

    try:
        results = await run_in_threadpool(
            get_retrieved_chunks,
            pipeline,
            req.question.strip(),
        )

        if not results:
            return {
                "valid": False,
                "message": "No evidence retrieved for verification.",
                "cited_sources": [],
                "invalid_sources": [],
            }

        result = verify_citations(req.answer, results)

        return {
            "question": req.question,
            "citation_check": result,
        }

    except Exception as error:
        handle_pipeline_error(error)

@app.post("/evaluate/answer")
async def evaluate_generated_answer(
    req: AnswerEvaluationRequest,
):
    pipeline = get_pipeline()

    try:
        results = await run_in_threadpool(
            get_retrieved_chunks,
            pipeline,
            req.question.strip(),
        )

        if not results:
            return {
                "question": req.question,
                "evaluation": None,
                "message": "No evidence retrieved for evaluation.",
            }

        evaluation = await run_in_threadpool(
            evaluate_answer,
            req.question,
            req.answer,
            results,
        )

        return {
            "question": req.question,
            "evaluation": evaluation,
        }

    except Exception as error:
        handle_pipeline_error(error)

@app.post("/papers")
async def papers(req: PapersRequest):
    pipeline = get_pipeline()

    try:
        return await run_in_threadpool(
            lambda: pipeline.find_papers(
                req.query.strip(),
                req.document_context,
                limit=req.limit,
                year_from=req.year_from,
                year_to=req.year_to,
                open_access_only=req.open_access_only,
                sort_by=req.sort_by,
            )
        )

    except Exception as error:
        handle_pipeline_error(error)

@app.post("/papers/search")
async def direct_paper_search(req: PapersRequest):
    try:
        return await run_in_threadpool(
            lambda: search_research_papers(
                user_query=req.query.strip(),
                document_context=req.document_context,
                limit=req.limit,
                year_from=req.year_from,
                year_to=req.year_to,
                open_access_only=req.open_access_only,
                sort_by=req.sort_by,
            )
        )

    except Exception as error:
        handle_pipeline_error(error)
