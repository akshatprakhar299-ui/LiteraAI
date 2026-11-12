from __future__ import annotations

import json
import re
from typing import Any

import requests

LLM_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

ALLOWED_VERDICTS = {
    "SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "UNCERTAIN",
}


def _normalise(text: str) -> str:
    """Normalize whitespace and case for exact quote matching."""
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _call_ollama(system_prompt: str, user_prompt: str) -> dict:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    response.raise_for_status()

    try:
        raw = response.json()["message"]["content"]
        parsed = json.loads(raw)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("The evaluator returned invalid JSON.") from error

    if not isinstance(parsed, dict):
        raise ValueError("Invalid evaluation structure.")

    return parsed


def _build_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    parts = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(
            f"[Evidence ID: {index}]\n"
            f"[Source: {chunk.get('source', '')}, Page: {chunk.get('page', '')}]\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(parts)


def _validate_claims(
    raw_claims: Any,
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(raw_claims, list):
        raise ValueError("Evaluator did not return a claims list.")

    validated = []

    for item in raw_claims:
        if not isinstance(item, dict):
            continue

        claim_text = str(item.get("claim", "")).strip()
        if not claim_text:
            continue

        verdict = str(item.get("verdict", "UNCERTAIN")).upper().strip()
        if verdict not in ALLOWED_VERDICTS:
            verdict = "UNCERTAIN"

        evidence = str(item.get("evidence", "")).strip()
        source = str(item.get("source", "")).strip()
        page = item.get("page")
        reason = str(item.get("reason", "")).strip()

        # A model-generated quote/reference is not trusted until it is
        # checked against the actual retrieved chunk text and metadata.
        matching_chunk = None
        if evidence:
            normalized_quote = _normalise(evidence)
            for chunk in retrieved_chunks:
                chunk_text = _normalise(chunk.get("text", ""))
                source_matches = (
                    not source
                    or source == str(chunk.get("source", ""))
                )
                page_matches = (
                    page is None
                    or str(page) == str(chunk.get("page", ""))
                )
                if (
                    normalized_quote
                    and normalized_quote in chunk_text
                    and source_matches
                    and page_matches
                ):
                    matching_chunk = chunk
                    break

        evidence_verified = matching_chunk is not None

        if verdict in {"SUPPORTED", "CONTRADICTED"} and not evidence_verified:
            verdict = "UNCERTAIN"
            reason = (
                "The evaluator did not provide an exact evidence quotation "
                "that could be matched to the retrieved document text. "
                "The original verdict was therefore downgraded."
            )

        if evidence_verified:
            source = str(matching_chunk.get("source", ""))
            page = matching_chunk.get("page")
        else:
            # Do not display an unverified source/page as if it were grounded.
            source = ""
            page = None
            evidence = ""

        validated.append({
            "claim": claim_text,
            "verdict": verdict,
            "evidence": evidence,
            "source": source,
            "page": page,
            "evidence_verified": evidence_verified,
            "reason": reason,
        })

    return validated


def _make_report(claims: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        verdict: sum(1 for claim in claims if claim["verdict"] == verdict)
        for verdict in ALLOWED_VERDICTS
    }

    total = len(claims)
    assessable = (
        counts["SUPPORTED"]
        + counts["UNSUPPORTED"]
        + counts["CONTRADICTED"]
    )
    faithfulness = (
        counts["SUPPORTED"] / assessable if assessable else None
    )

    # This is an evidence-coverage indicator, not a calibrated probability.
    verified_supported = sum(
        1 for claim in claims
        if claim["verdict"] == "SUPPORTED" and claim["evidence_verified"]
    )
    coverage = verified_supported / total if total else None

    if not total:
        confidence = "NOT ASSESSED"
    elif (
        counts["UNSUPPORTED"] > 0
        or counts["CONTRADICTED"] > 0
        or counts["UNCERTAIN"] > 0
    ):
        confidence = "LOW"
    elif coverage is not None and coverage >= 0.8:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    return {
        "total_claims": total,
        "supported": counts["SUPPORTED"],
        "unsupported": counts["UNSUPPORTED"],
        "contradicted": counts["CONTRADICTED"],
        "uncertain": counts["UNCERTAIN"],
        "faithfulness": round(faithfulness, 4) if faithfulness is not None else None,
        "evidence_coverage": round(coverage, 4) if coverage is not None else None,
        "confidence": confidence,
        "claims": claims,
        "evaluation_method": "LLM evidence judgment with exact-quote validation",
        "warning": (
            "Automated evaluation is an aid, not a guarantee of factual correctness. "
            "Confidence is a heuristic evidence-coverage label, not a probability."
        ),
    }


def evaluate_answer(
    question: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Break an answer into factual claims and judge each claim against
    retrieved PDF passages. Any supported/contradicted verdict must include
    an exact quotation that is subsequently matched to retrieved text.
    """
    if not retrieved_chunks:
        return _make_report([{
            "claim": "The answer could not be checked because no evidence was retrieved.",
            "verdict": "UNCERTAIN",
            "evidence": "",
            "source": "",
            "page": None,
            "evidence_verified": False,
            "reason": "No retrieved document evidence was available.",
        }])

    system_prompt = """
You are a strict evidence-verification evaluator for a document question-answering system.
Use ONLY the supplied retrieved document passages. Do not use outside knowledge.

Split the answer into atomic factual claims. Evaluate every factual claim independently:
SUPPORTED: directly supported by the supplied passages.
UNSUPPORTED: not supported by the supplied passages.
CONTRADICTED: directly conflicts with a supplied passage.
UNCERTAIN: evidence is ambiguous or insufficient.

Important rules:
- A citation label by itself is never evidence.
- For SUPPORTED or CONTRADICTED, provide an exact, continuous quotation copied verbatim
  from one supplied passage. Do not paraphrase the quotation.
- Copy the exact Evidence ID, source filename, and page associated with that quotation.
- If you cannot provide an exact quotation, use UNCERTAIN.
- Do not invent page numbers, filenames, quotations, facts, or experiments.
- Ignore conversational filler and opinions.
- If the answer explicitly abstains because information is absent, do not count that
  abstention as a factual claim.
- Return ONLY valid JSON in this shape:
{
  "claims": [
    {
      "claim": "Atomic factual claim",
      "verdict": "SUPPORTED",
      "evidence": "Exact quotation copied from the passage",
      "source": "filename.pdf",
      "page": 1,
      "reason": "Short explanation"
    }
  ]
}
If there are no factual claims, return {"claims": []}.
"""

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        f"RETRIEVED DOCUMENT EVIDENCE:\n{_build_context(retrieved_chunks)}\n\n"
        "Evaluate every factual claim independently. Return only JSON."
    )

    raw = _call_ollama(system_prompt, user_prompt)
    claims = _validate_claims(raw.get("claims"), retrieved_chunks)
    return _make_report(claims)


def correct_answer(
    question: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> str:
    """
    Make one conservative correction pass. The model must use only retrieved
    evidence and explicitly abstain where evidence is insufficient.
    """
    problematic = [
        claim for claim in evaluation.get("claims", [])
        if claim.get("verdict") in {"UNSUPPORTED", "CONTRADICTED", "UNCERTAIN"}
    ]

    if not problematic:
        return answer

    system_prompt = """
You are revising a document-grounded answer to remove unsupported statements.
Use ONLY the supplied document evidence.
- Keep claims only when directly supported by the passages.
- Remove unsupported or contradicted claims.
- If a requested detail is absent, clearly say it is not stated in the provided documents.
- Do not infer missing facts or use outside knowledge.
- Add a source filename and page reference for each factual statement, using only
  source/page details supplied in the evidence.
- Never invent quotations, citations, numbers, experiments, names, or dates.
- Return only the revised answer, with no commentary about the revision.
"""

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"ORIGINAL ANSWER:\n{answer}\n\n"
        f"CLAIMS REQUIRING CORRECTION:\n"
        f"{json.dumps(problematic, ensure_ascii=False, indent=2)}\n\n"
        f"ALLOWED DOCUMENT EVIDENCE:\n{_build_context(retrieved_chunks)}\n\n"
        "Rewrite the answer conservatively using only the allowed evidence."
    )

    result = _call_ollama(system_prompt, user_prompt)
    revised = result.get("answer") or result.get("revised_answer")
    if not isinstance(revised, str) or not revised.strip():
        raise ValueError("The answer correction step returned no answer.")
    return revised.strip()


def display_evaluation(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("       CLAIM-LEVEL EVIDENCE ANALYSIS")
    print("=" * 60)
    print(f"\nTotal claims: {report['total_claims']}")
    print(f"Supported: {report['supported']}")
    print(f"Unsupported: {report['unsupported']}")
    print(f"Contradicted: {report['contradicted']}")
    print(f"Uncertain: {report['uncertain']}")
    print(f"Confidence: {report['confidence']}")

    if report["faithfulness"] is not None:
        print(f"Faithfulness: {report['faithfulness']:.1%}")
    else:
        print("Faithfulness: Not measurable")

    for number, claim in enumerate(report["claims"], start=1):
        print(f"\nClaim {number}: {claim['claim']}")
        print(f"Verdict: {claim['verdict']}")
        print(f"Evidence verified: {claim['evidence_verified']}")
        print(f"Evidence: {claim['evidence']}")
        print(f"Source: {claim['source']} | Page: {claim['page']}")
        print(f"Reason: {claim['reason']}")

    print("\n" + report["warning"])


if __name__ == "__main__":
    print("This module is designed to be called from the RAG pipeline.")
