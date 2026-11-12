"""
RAG Doctor - Research Paper Discovery

Searches scholarly works using OpenAlex and ranks candidate papers
using the user's request plus context extracted from uploaded PDFs.

API key setup:
    1. Install dependencies:
       pip install requests python-dotenv

    2. Add this to the project-root .env file:
       OPENALEX_API_KEY=your_key_here

    3. Add .env to .gitignore.

OpenAlex API documentation:
    https://docs.openalex.org/

The API key is optional for basic testing, but authenticated requests
receive the account's applicable usage allowance.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load .env from the project root (one directory above this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
OPENALEX_WORKS_URL = "https://api.openalex.org/works"

REQUEST_TIMEOUT = 30
DEFAULT_LIMIT = 10
MAX_LIMIT = 25

# Optional: identify your application to OpenAlex.
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()

STOP_WORDS = {
    "about", "above", "after", "again", "against", "also", "among",
    "and", "any", "are", "because", "been", "before", "being", "below",
    "between", "both", "but", "can", "could", "did", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "here", "how", "into", "its", "itself",
    "just", "more", "most", "not", "now", "only", "other", "our",
    "out", "over", "same", "should", "some", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "under", "until", "very", "was", "were", "what",
    "when", "where", "which", "while", "who", "why", "will", "with",
    "would", "you", "your", "paper", "research", "study", "studies",
    "using", "used", "based", "approach", "method", "methods",
}

def extract_context_keywords(
    text: str,
    max_keywords: int = 8,
) -> List[str]:
    """
    Extract frequent, meaningful terms from uploaded PDF text.

    This is lightweight keyword expansion, not semantic topic modelling.
    """
    if not text:
        return []

    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    words = [
        word for word in words
        if word not in STOP_WORDS and not word.isdigit()
    ]

    counts = Counter(words)
    return [word for word, _ in counts.most_common(max_keywords)]


def build_search_query(
    user_query: str,
    document_context: Optional[str] = None,
    max_context_keywords: int = 8,
) -> str:
    """
    Keep the user's explicit request primary, then add selected terms
    extracted from the uploaded document context.
    """
    user_query = (user_query or "").strip()

    if not user_query:
        raise ValueError(
            "Please describe the kind of research paper you want."
        )

    context_keywords = extract_context_keywords(
        document_context or "",
        max_keywords=max_context_keywords,
    )

    existing_terms = set(
        re.findall(r"[a-z0-9-]+", user_query.lower())
    )
    additional_terms = [
        term for term in context_keywords
        if term not in existing_terms
    ]

    combined_query = user_query
    if additional_terms:
        combined_query = f"{user_query} {' '.join(additional_terms)}"

    return combined_query[:700]


def reconstruct_abstract(
    inverted_index: Optional[Dict[str, List[int]]],
) -> str:
    """Reconstruct an abstract from OpenAlex inverted-index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    positioned_words = []

    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue

        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned_words.append((position, word))

    positioned_words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned_words)


def _get_author_names(
    authorships: List[Dict[str, Any]],
) -> List[str]:
    authors = []

    for authorship in authorships or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")

        if name:
            authors.append(name)

    return authors


def _extract_locations(
    work: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    """Extract available open-access PDF and landing-page URLs."""
    best_oa = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}

    pdf_url = best_oa.get("pdf_url")

    landing_page_url = (
        best_oa.get("landing_page_url")
        or primary.get("landing_page_url")
        or work.get("doi")
    )

    source = (
        (primary.get("source") or {}).get("display_name")
        or (best_oa.get("source") or {}).get("display_name")
    )

    return {
        "pdf_url": pdf_url,
        "landing_page_url": landing_page_url,
        "journal_or_repository": source,
    }

def _normalize_work(work: Dict[str, Any]) -> Dict[str, Any]:
    locations = _extract_locations(work)
    open_access = work.get("open_access") or {}

    return {
        "openalex_id": work.get("id"),
        "title": work.get("display_name") or "Untitled work",
        "authors": _get_author_names(work.get("authorships") or []),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "abstract": reconstruct_abstract(
            work.get("abstract_inverted_index")
        ),
        "doi": work.get("doi"),
        "landing_page_url": locations["landing_page_url"],
        "pdf_url": locations["pdf_url"],
        "journal_or_repository": locations["journal_or_repository"],
        "cited_by_count": work.get("cited_by_count", 0),
        "is_open_access": bool(open_access.get("is_oa", False)),
        "open_access_status": open_access.get("oa_status"),
        "relevance_score": work.get("relevance_score"),
    }



def _build_headers() -> Dict[str, str]:
    return {
        "User-Agent": "RAG-Doctor/1.0 (research paper discovery)"
    }


def _build_base_params() -> Dict[str, Any]:
    """
    OpenAlex accepts the API key as an api_key query parameter.
    The key is never printed or returned in result dictionaries.
    """
    params: Dict[str, Any] = {}

    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    return params


def check_openalex_authentication() -> Dict[str, Any]:
    """
    Check that OpenAlex is reachable and, if configured, that the
    supplied API key is accepted.
    """
    params = {
        **_build_base_params(),
        "search": "retrieval augmented generation",
        "per-page": 1,
        "select": "id,display_name",
    }

    try:
        response = requests.get(
            OPENALEX_WORKS_URL,
            params=params,
            headers=_build_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()

        return {
            "success": True,
            "authenticated_key_configured": bool(OPENALEX_API_KEY),
            "message": (
                "OpenAlex connection and API key check successful."
                if OPENALEX_API_KEY
                else "OpenAlex connection successful. No API key is configured."
            ),
            "sample_result_count": len(payload.get("results", [])),
        }

    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None

        if status in (401, 403):
            message = (
                "OpenAlex rejected the API key. Check that it is correct "
                "and activated in your OpenAlex account."
            )
        else:
            message = f"OpenAlex returned HTTP {status}: {exc}"

        return {
            "success": False,
            "authenticated_key_configured": bool(OPENALEX_API_KEY),
            "message": message,
        }

    except requests.RequestException as exc:
        return {
            "success": False,
            "authenticated_key_configured": bool(OPENALEX_API_KEY),
            "message": f"Could not connect to OpenAlex: {exc}",
        }

    except ValueError:
        return {
            "success": False,
            "authenticated_key_configured": bool(OPENALEX_API_KEY),
            "message": "OpenAlex returned an invalid JSON response.",
        }


# ---------------------------------------------------------------------
# OPENALEX SEARCH
# ---------------------------------------------------------------------

def search_research_papers(
    user_query: str,
    document_context: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    open_access_only: bool = False,
    sort_by: str = "relevance",
) -> Dict[str, Any]:
    """
    Search OpenAlex for research papers related to the user's request
    and optional context extracted from uploaded PDF(s).

    Args:
        user_query:
            User's explicit research request.
        document_context:
            Bounded text sample or summary from the user's uploaded PDFs.
        limit:
            Number of results (1-25).
        year_from / year_to:
            Optional publication-year boundaries.
        open_access_only:
            If True, restrict results to OpenAlex open-access works.
        sort_by:
            "relevance", "newest", or "cited".

    Returns:
        Search metadata and normalized paper records.
    """
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")

    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer.")

    if year_from is not None:
        year_from = int(year_from)

    if year_to is not None:
        year_to = int(year_to)

    if (
        year_from is not None
        and year_to is not None
        and year_from > year_to
    ):
        raise ValueError("year_from cannot be later than year_to.")

    if sort_by not in {"relevance", "newest", "cited"}:
        raise ValueError(
            "sort_by must be 'relevance', 'newest', or 'cited'."
        )

    search_query = build_search_query(
        user_query=user_query,
        document_context=document_context,
    )

    params: Dict[str, Any] = {
        **_build_base_params(),
        "search": search_query,
        "per-page": limit,
        "select": (
            "id,doi,display_name,publication_year,publication_date,"
            "type,authorships,primary_location,best_oa_location,"
            "open_access,cited_by_count,abstract_inverted_index,"
            "relevance_score"
        ),
    }

    filters = []

    if year_from is not None:
        filters.append(
            f"from_publication_date:{year_from}-01-01"
        )

    if year_to is not None:
        filters.append(
            f"to_publication_date:{year_to}-12-31"
        )

    if open_access_only:
        filters.append("open_access.is_oa:true")

    if filters:
        params["filter"] = ",".join(filters)

    if sort_by == "newest":
        params["sort"] = "publication_date:desc"
    elif sort_by == "cited":
        params["sort"] = "cited_by_count:desc"

    try:
        response = requests.get(
            OPENALEX_WORKS_URL,
            params=params,
            headers=_build_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()

    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None

        if status in (401, 403):
            raise RuntimeError(
                "OpenAlex rejected the API key. Check OPENALEX_API_KEY "
                "in your project-root .env file and verify that the key "
                "is active."
            ) from exc

        if status == 429:
            raise RuntimeError(
                "OpenAlex rate limit reached. Wait before retrying."
            ) from exc

        raise RuntimeError(
            f"OpenAlex returned HTTP {status}: {exc}"
        ) from exc

    except requests.Timeout as exc:
        raise RuntimeError(
            "OpenAlex request timed out. Please try again."
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            f"OpenAlex research search failed: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "OpenAlex returned an invalid JSON response."
        ) from exc

    raw_results = payload.get("results", [])
    papers = [_normalize_work(work) for work in raw_results]

    return {
        "provider": "OpenAlex",
        "authenticated_key_configured": bool(OPENALEX_API_KEY),
        "user_query": user_query,
        "search_query": search_query,
        "total_matching_works": (
            payload.get("meta") or {}
        ).get("count"),
        "returned": len(papers),
        "filters": {
            "year_from": year_from,
            "year_to": year_to,
            "open_access_only": open_access_only,
            "sort_by": sort_by,
        },
        "papers": papers,
        "notice": (
            "Results contain scholarly-work metadata. A direct PDF link "
            "is included only when OpenAlex supplies one. Otherwise use "
            "the landing page or DOI. Check publisher access and license "
            "terms before downloading or redistributing full text."
        ),
    }

def display_papers(report: Dict[str, Any]) -> None:
    """Print concise research search results."""
    print("\n" + "=" * 72)
    print("RAG DOCTOR - RESEARCH PAPER DISCOVERY")
    print("=" * 72)

    print(f"User request: {report['user_query']}")
    print(f"Search query: {report['search_query']}")
    print(f"Results returned: {report['returned']}")

    for number, paper in enumerate(report["papers"], start=1):
        authors = ", ".join(paper["authors"][:5]) or "Authors unavailable"

        if len(paper["authors"]) > 5:
            authors += " et al."

        print(f"\n{number}. {paper['title']}")
        print(f"   Authors: {authors}")
        print(f"   Year: {paper['publication_year'] or 'Unknown'}")
        print(f"   Type: {paper['type'] or 'Unknown'}")
        print(f"   Citations: {paper['cited_by_count']}")
        print(f"   Open access: {paper['is_open_access']}")
        print(f"   DOI: {paper['doi'] or 'Not available'}")
        print(f"   PDF: {paper['pdf_url'] or 'No direct PDF link'}")
        print(
            f"   Page: "
            f"{paper['landing_page_url'] or 'Not available'}"
        )

        if paper["abstract"]:
            abstract = paper["abstract"]
            preview = abstract[:450].strip()
            suffix = "..." if len(abstract) > 450 else ""
            print(f"   Abstract: {preview}{suffix}")

    print("\n" + report["notice"])


def main() -> None:
    print("RAG Doctor - Research Paper Discovery")

    auth = check_openalex_authentication()
    print(auth["message"])

    if not auth["success"]:
        print("Check your internet connection and API key configuration.")
        return

    query = input(
        "\nWhat kind of research papers are you looking for? "
    ).strip()

    context = input(
        "Paste a short topic summary from the uploaded PDF (optional): "
    ).strip()

    try:
        report = search_research_papers(
            user_query=query,
            document_context=context or None,
            limit=10,
            open_access_only=False,
            sort_by="relevance",
        )
        display_papers(report)

    except (ValueError, RuntimeError) as error:
        print(f"\nResearch search error: {error}")


if __name__ == "__main__":
    main()
