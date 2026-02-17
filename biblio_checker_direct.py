#!/usr/bin/env python3
"""biblio_checker_direct.py — Verify paper references without LLM. $0 cost.

Reads a draft markdown, extracts References section, verifies each against
CrossRef (DOI) and Semantic Scholar (arXiv/title search). Pure Python.

Usage:
    biblio_checker_direct.py --pub-id PUB-012 --draft path/to/DRAFT.md [--output-dir dir] [--json]
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("biblio-checker")

# S2 API key
_ENV_FILE = Path.home() / ".env.api_keys"
S2_API_KEY = ""
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if line.startswith("SEMANTIC_SCHOLAR_API_KEY="):
            S2_API_KEY = line.split("=", 1)[1].strip()
            break

# Rate limiting
_LAST_S2_CALL = 0.0
_S2_INTERVAL = 1.1 if S2_API_KEY else 3.5  # 1 req/s with key, ~20/min without


def _s2_throttle():
    global _LAST_S2_CALL
    elapsed = time.time() - _LAST_S2_CALL
    if elapsed < _S2_INTERVAL:
        time.sleep(_S2_INTERVAL - elapsed)
    _LAST_S2_CALL = time.time()


def _http_get(url: str, headers: dict | None = None) -> dict | None:
    """GET JSON from URL. Returns None on any error. Logs errors in debug mode."""
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("User-Agent", "biblio-checker/1.0 (mailto:jhdez32@gmail.com)")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        log.debug("HTTP GET %s → %s: %s", url[:80], type(exc).__name__, exc)
        return None


def _normalize(text: str) -> str:
    """Lowercase, replace all dashes with spaces, strip punctuation, collapse ws."""
    text = text.lower()
    # Replace ASCII hyphen + common Unicode dashes with space
    text = re.sub(r"[\u002D\u2010\u2011\u2012\u2013\u2014\u2212]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_match(a: str, b: str) -> bool:
    """Fuzzy title match: normalized containment or >85% overlap."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    # One contains the other (handles subtitle differences)
    if na in nb or nb in na:
        return True
    # Word overlap
    wa, wb = set(na.split()), set(nb.split())
    if not wa or not wb:
        return False
    overlap = len(wa & wb) / max(len(wa), len(wb))
    return overlap > 0.85


# --- Extraction ---

def extract_references(text: str) -> list[dict]:
    """Parse numbered references from markdown."""
    refs = []
    # Find References section (English/Spanish variants)
    match = re.search(r"^##\s+(?:References|Referencias|Bibliography|Bibliograf[ií]a)\s*$",
                       text, re.MULTILINE | re.IGNORECASE)
    if not match:
        return refs
    ref_text = text[match.end():]
    # Stop at next section or end marker
    end = re.search(r"^---\s*$|^## ", ref_text, re.MULTILINE)
    if end:
        ref_text = ref_text[:end.start()]

    # Parse numbered entries: "N. text" or "[N] text" (handles multi-line)
    entries = re.split(r"\n(?=\d+\.\s|\[\d+\]\s)", ref_text.strip())
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        num_match = re.match(r"(?:(\d+)\.\s+|\[(\d+)\]\s+)(.+)", entry, re.DOTALL)
        if not num_match:
            continue
        num = int(num_match.group(1) or num_match.group(2))
        body = num_match.group(3).replace("\n", " ").strip()

        doi = None
        arxiv = None
        isbn = None
        # DOI: URL form (doi.org/...) or inline form (doi:10.xxxx/... or DOI: 10.xxxx/...)
        # Greedy match excluding delimiters (NOT dots — dots are valid in DOIs)
        doi_match = re.search(
            r"(?:https?://doi\.org/|(?:doi|DOI)[:\s]+)(10\.\d{4,9}/[^\s,;)\]]+)", body)
        if doi_match:
            doi = doi_match.group(1).rstrip(".")
        # arXiv: standard (arXiv:NNNN.NNNNN), legacy (arXiv:hep-ph/NNNNNNN), case-insensitive
        arxiv_match = re.search(
            r"(?i)arxiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)", body)
        if arxiv_match:
            arxiv = arxiv_match.group(1)
        isbn_match = re.search(r"ISBN[:\s]*([\d-]+)", body)
        if isbn_match:
            isbn = isbn_match.group(1)

        # Extract title (between first period after authors and next *)
        title = _extract_title(body)

        refs.append({
            "ref_num": num,
            "raw": body,
            "title": title,
            "doi": doi,
            "arxiv": arxiv,
            "isbn": isbn,
        })
    return refs


def _extract_title(body: str) -> str:
    """Best-effort title extraction from citation string."""
    # Pattern: Authors (year). Title. *Venue*...
    # or: Authors (year). *Title*. Venue...
    m = re.search(r"\(\d{4}\)\.\s*\*?(.+?)\*?\.\s*\*", body)
    if m:
        return m.group(1).strip().rstrip(".")
    # Fallback: text between ). and first *
    m = re.search(r"\)\.\s+(.+?)\.\s+\*", body)
    if m:
        return m.group(1).strip()
    # Last resort: everything after (year).
    m = re.search(r"\(\d{4}\)\.\s+(.+?)(?:\.|$)", body)
    if m:
        return m.group(1).strip()
    return body[:80]


# --- Verification ---

def verify_by_doi(doi: str) -> dict:
    """Verify via CrossRef."""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    data = _http_get(url)
    if not data or "message" not in data:
        return {"source": "crossref", "found": False}
    msg = data["message"]
    title = msg.get("title", [""])[0]
    year = None
    if "published-print" in msg:
        year = msg["published-print"].get("date-parts", [[None]])[0][0]
    elif "published-online" in msg:
        year = msg["published-online"].get("date-parts", [[None]])[0][0]
    elif "created" in msg:
        year = msg["created"].get("date-parts", [[None]])[0][0]
    authors = []
    for a in msg.get("author", [])[:3]:
        name = f"{a.get('given', '')} {a.get('family', '')}".strip()
        if name:
            authors.append(name)
    return {"source": "crossref", "found": True, "title": title, "year": year,
            "authors": authors, "doi": doi}


def verify_by_arxiv(arxiv_id: str) -> dict:
    """Verify via Semantic Scholar arXiv endpoint."""
    _s2_throttle()
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=title,authors,year,externalIds"
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    data = _http_get(url, headers)
    if not data or "title" not in data:
        return {"source": "s2-arxiv", "found": False}
    authors = [a.get("name", "") for a in data.get("authors", [])[:3]]
    doi = (data.get("externalIds") or {}).get("DOI")
    return {"source": "s2-arxiv", "found": True, "title": data["title"],
            "year": data.get("year"), "authors": authors, "doi": doi}


def verify_by_title(title: str) -> dict:
    """Search Semantic Scholar by title."""
    _s2_throttle()
    q = urllib.parse.quote(title[:200])
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=3&fields=title,authors,year,externalIds"
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    data = _http_get(url, headers)
    if not data or not data.get("data"):
        return {"source": "s2-search", "found": False}
    for paper in data["data"]:
        if _title_match(title, paper.get("title", "")):
            authors = [a.get("name", "") for a in paper.get("authors", [])[:3]]
            doi = (paper.get("externalIds") or {}).get("DOI")
            return {"source": "s2-search", "found": True, "title": paper["title"],
                    "year": paper.get("year"), "authors": authors, "doi": doi}
    return {"source": "s2-search", "found": False}


def verify_ref(ref: dict) -> dict:
    """Verify a single reference. Returns result dict."""
    result = {"ref_num": ref["ref_num"], "title_draft": ref["title"]}

    # Books
    if ref.get("isbn") and not ref.get("doi") and not ref.get("arxiv"):
        result["status"] = "UNVERIFIABLE"
        result["notes"] = f"Book (ISBN: {ref['isbn']})"
        return result

    # Try DOI first
    if ref.get("doi"):
        api = verify_by_doi(ref["doi"])
        if api["found"]:
            result["title_api"] = api["title"]
            result["doi"] = ref["doi"]
            result["year_api"] = api.get("year")
            if _title_match(ref["title"], api["title"]):
                result["status"] = "VERIFIED"
                result["notes"] = "DOI verified via CrossRef"
            else:
                result["status"] = "MISMATCH"
                result["notes"] = f"Title mismatch: draft='{ref['title'][:60]}' vs api='{api['title'][:60]}'"
            return result

    # Try arXiv
    if ref.get("arxiv"):
        api = verify_by_arxiv(ref["arxiv"])
        if api["found"]:
            result["title_api"] = api["title"]
            result["doi"] = api.get("doi")
            result["year_api"] = api.get("year")
            if _title_match(ref["title"], api["title"]):
                result["status"] = "VERIFIED"
                result["notes"] = "arXiv verified via Semantic Scholar"
            else:
                result["status"] = "MISMATCH"
                result["notes"] = f"Title mismatch: draft='{ref['title'][:60]}' vs api='{api['title'][:60]}'"
            return result

    # Fallback: title search
    api = verify_by_title(ref["title"])
    if api["found"]:
        result["title_api"] = api["title"]
        result["doi"] = api.get("doi")
        result["year_api"] = api.get("year")
        result["status"] = "VERIFIED"
        result["notes"] = "Title match via S2 search"
        return result

    result["status"] = "NOT_FOUND"
    result["notes"] = "No match in CrossRef or Semantic Scholar"
    return result


# --- Output ---

def write_report(results: list[dict], output_path: str, pub_id: str):
    """Write markdown + JSON report."""
    counts = {"VERIFIED": 0, "MISMATCH": 0, "NOT_FOUND": 0, "UNVERIFIABLE": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    lines = [
        f"# Reference Verification Report — {pub_id}\n",
        f"**Total:** {len(results)} | "
        f"**Verified:** {counts['VERIFIED']} | "
        f"**Mismatch:** {counts['MISMATCH']} | "
        f"**Not Found:** {counts['NOT_FOUND']} | "
        f"**Unverifiable:** {counts['UNVERIFIABLE']}\n",
        "| # | Status | Title | DOI/arXiv | Notes |",
        "|---|--------|-------|-----------|-------|",
    ]
    for r in results:
        doi_str = r.get("doi") or ""
        title_short = r.get("title_draft", "")[:50]
        lines.append(f"| {r['ref_num']} | {r['status']} | {title_short} | {doi_str} | {r.get('notes', '')} |")

    summary = {
        "total": len(results),
        "verified": counts["VERIFIED"],
        "mismatch": counts["MISMATCH"],
        "not_found": counts["NOT_FOUND"],
        "unverifiable": counts["UNVERIFIABLE"],
        "references": results,
    }
    lines.append("\n```json")
    lines.append(json.dumps(summary, indent=2, ensure_ascii=False))
    lines.append("```\n")

    Path(output_path).write_text("\n".join(lines))
    return summary


def _escalate_to_llm(unresolved: list[dict], draft_path: str, output_dir: str,
                      model: str) -> tuple[list[dict], float]:
    """Pass unresolved refs to Claude Code SDK agent for deeper analysis."""
    import asyncio
    try:
        from claude_code_sdk import query as sdk_query, ClaudeCodeOptions, ResultMessage
    except ImportError:
        print("  SDK not installed, skipping LLM escalation", file=sys.stderr)
        return unresolved, 0.0

    os.environ.pop("CLAUDECODE", None)

    ref_block = "\n".join(
        f"- Ref #{r['ref_num']}: status={r['status']}, "
        f"title=\"{r.get('title_draft', '')}\", "
        f"doi={r.get('doi', 'none')}, notes={r.get('notes', '')}"
        for r in unresolved
    )

    prompt = (
        f"I ran automated reference verification on {draft_path}. "
        f"These {len(unresolved)} references need deeper analysis "
        f"(author verification, title variant matching, etc.):\n\n"
        f"{ref_block}\n\n"
        f"For each reference:\n"
        f"1. Search CrossRef/Semantic Scholar to find the correct paper\n"
        f"2. Compare authors (not just titles) against what's in the draft\n"
        f"3. Determine: VERIFIED, MISMATCH (explain what's wrong), or NOT_FOUND\n\n"
        f"Read the draft at {draft_path} to see the full citation text.\n"
        f"Write results to {output_dir}/REFERENCES_LLM_REVIEW.md\n\n"
        f"End with a JSON code block:\n"
        f"```json\n"
        f'[{{"ref_num": N, "status": "...", "title_draft": "...", '
        f'"title_api": "...", "notes": "..."}}]\n'
        f"```"
    )

    env = {}
    if S2_API_KEY:
        env["SEMANTIC_SCHOLAR_API_KEY"] = S2_API_KEY

    options = ClaudeCodeOptions(
        system_prompt=(
            "You are a bibliographic reference auditor. You verify paper references "
            "by checking CrossRef and Semantic Scholar APIs via WebFetch. "
            "Compare titles AND authors. Be precise about what mismatches."
        ),
        allowed_tools=["WebFetch", "Read", "Write"],
        model=model,
        max_turns=15,
        permission_mode="acceptEdits",
        env=env,
    )

    async def _run():
        cost = 0.0
        result_text = ""
        try:
            async for msg in sdk_query(prompt=prompt, options=options):
                if isinstance(msg, ResultMessage):
                    cost = msg.total_cost_usd or 0.0
                    result_text = msg.result or ""
        except Exception as exc:
            print(f"  LLM escalation failed: {exc}", file=sys.stderr)
            return unresolved, cost
        return result_text, cost

    result_text, cost = asyncio.run(_run())
    if isinstance(result_text, list):
        # _run returned unresolved unchanged on error
        return unresolved, cost

    # Parse LLM results
    llm_results = _extract_llm_refs(result_text)

    # Also try from files the agent may have written
    if not llm_results:
        for fname in ("REFERENCES_LLM_REVIEW.md", "REFERENCE_VERIFICATION_RESULTS.json"):
            fpath = Path(output_dir) / fname
            if fpath.exists():
                content = fpath.read_text()
                llm_results = _extract_llm_refs(content)
                if llm_results:
                    break

    if not llm_results:
        print(f"  LLM returned no parseable results (cost: ${cost:.4f})", file=sys.stderr)
        return unresolved, cost

    # Merge: LLM results override unresolved, normalize statuses
    _STATUS_MAP = {
        "VERIFIED": "VERIFIED", "MISMATCH": "MISMATCH",
        "NOT_FOUND": "NOT_FOUND", "UNVERIFIABLE": "UNVERIFIABLE",
    }
    llm_by_num = {r["ref_num"]: r for r in llm_results}
    merged = []
    for r in unresolved:
        if r["ref_num"] in llm_by_num:
            updated = llm_by_num[r["ref_num"]]
            raw_status = updated.get("status", "NOT_FOUND")
            # Normalize non-standard statuses: MISMATCH_TITLE_VARIANT → MISMATCH
            norm = _STATUS_MAP.get(raw_status)
            if norm is None:
                for prefix in ("MISMATCH", "NOT_FOUND", "VERIFIED", "UNVERIFIABLE"):
                    if raw_status.startswith(prefix):
                        norm = prefix
                        break
                else:
                    norm = "NOT_FOUND"
            detail = raw_status if raw_status != norm else ""
            updated["status"] = norm
            updated["notes"] = f"[LLM] {detail} {updated.get('notes', '')}".strip()
            merged.append(updated)
        else:
            merged.append(r)

    print(f"  LLM reviewed {len(llm_results)} refs (cost: ${cost:.4f})")
    return merged, cost


def _reality_check(results: list[dict]) -> list[dict]:
    """Phase 3: Re-verify LLM claims against real APIs.

    For each ref tagged [LLM], if LLM claims VERIFIED and provides a DOI or
    arXiv ID, hit the same APIs Python uses to confirm. If the API contradicts
    the LLM, downgrade the status. Zero trust on LLM assertions.
    """
    checked = []
    n_checked = 0
    n_downgraded = 0
    for r in results:
        notes = r.get("notes", "")
        if "[LLM]" not in notes:
            checked.append(r)
            continue

        status = r.get("status", "NOT_FOUND")
        doi = r.get("doi")
        arxiv_id = r.get("arxiv_id")
        title = r.get("title_draft", "")

        # Only re-check if LLM claims VERIFIED or provides identifiers
        if status == "VERIFIED" or doi or arxiv_id:
            n_checked += 1
            api_result = None

            # Try DOI first (skip arXiv DOIs initially — prefer arXiv endpoint)
            if doi and not doi.startswith("10.48550"):
                api_result = verify_by_doi(doi)

            # Try arXiv
            if (api_result is None or not api_result.get("found")) and arxiv_id:
                api_result = verify_by_arxiv(arxiv_id)

            # Try arXiv DOI as fallback (10.48550 skipped initially)
            if (api_result is None or not api_result.get("found")) and doi and doi.startswith("10.48550"):
                api_result = verify_by_doi(doi)

            # Try title search as last resort
            if (api_result is None or not api_result.get("found")) and title and status == "VERIFIED":
                api_result = verify_by_title(title)

            if api_result and api_result.get("found"):
                # API found something — confirm or downgrade
                api_title = api_result.get("title", "")
                if api_title and title and _title_match(title, api_title):
                    r["title_api"] = api_title
                    r["doi"] = r.get("doi") or api_result.get("doi")
                    r["year_api"] = api_result.get("year")
                    r["notes"] = notes + " [API-confirmed]"
                elif api_title:
                    # API returned something but title doesn't match
                    if status == "VERIFIED":
                        r["status"] = "MISMATCH"
                        n_downgraded += 1
                    r["title_api"] = api_title
                    r["notes"] = notes + f" [API-downgraded: title mismatch '{api_title[:60]}']"
            else:
                # API found nothing — if LLM said VERIFIED, downgrade
                if status == "VERIFIED":
                    r["status"] = "NOT_FOUND"
                    n_downgraded += 1
                    r["notes"] = notes + " [API-downgraded: not found in any API]"

        checked.append(r)

    if n_checked > 0:
        print(f"  Reality check: {n_checked} LLM claims verified, {n_downgraded} downgraded")
    return checked


def _extract_llm_refs(text: str) -> list[dict]:
    """Extract ref list from LLM output (fenced JSON, raw JSON array, or dict)."""
    # Try fenced code block
    if "```json" in text:
        try:
            start = text.index("```json") + 7
            end = text.index("```", start)
            data = json.loads(text[start:end].strip())
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "references" in data:
                return data["references"]
        except (json.JSONDecodeError, ValueError):
            pass
    # Try raw JSON (whole text is valid JSON)
    text_stripped = text.strip()
    if text_stripped.startswith("[") or text_stripped.startswith("{"):
        try:
            data = json.loads(text_stripped)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "references" in data:
                return data["references"]
        except json.JSONDecodeError:
            pass
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Verify paper references. Python direct ($0) + optional LLM escalation."
    )
    parser.add_argument("--pub-id", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--escalate", action="store_true",
                        help="Pass NOT_FOUND/MISMATCH refs to LLM for deeper analysis")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Model for LLM escalation (default: haiku)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (shows HTTP errors, API responses)")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

    draft = Path(args.draft).resolve()
    if not draft.exists():
        print(f"Error: {draft}", file=sys.stderr)
        sys.exit(2)

    text = draft.read_text()
    refs = extract_references(text)
    if not refs:
        print("No references found in draft.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else draft.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / "REFERENCES_VERIFIED.md")

    # Phase 1: Python direct ($0)
    t0 = time.time()
    results = []
    for i, ref in enumerate(refs):
        results.append(verify_ref(ref))
        if not args.json_output:
            r = results[-1]
            print(f"  [{i+1}/{len(refs)}] #{r['ref_num']} {r['status']}: {r.get('title_draft', '')[:50]}")

    elapsed_direct = time.time() - t0
    unresolved = [r for r in results if r["status"] in ("NOT_FOUND", "MISMATCH")]

    # Phase 2: LLM escalation (only if --escalate and there are unresolved)
    cost_usd = 0.0
    if args.escalate and unresolved:
        if not args.json_output:
            print(f"\n  Escalating {len(unresolved)} unresolved refs to LLM...")
        escalated, cost_usd = _escalate_to_llm(unresolved, str(draft), str(out_dir), args.model)
        # Replace unresolved in results
        esc_by_num = {r["ref_num"]: r for r in escalated}
        for i, r in enumerate(results):
            if r["ref_num"] in esc_by_num:
                results[i] = esc_by_num[r["ref_num"]]

        # Phase 3: Reality check — verify LLM claims against real APIs
        results = _reality_check(results)

    elapsed_total = time.time() - t0
    summary = write_report(results, output_path, args.pub_id)
    summary["_meta"] = {
        "elapsed_s": round(elapsed_total, 1),
        "direct_s": round(elapsed_direct, 1),
        "cost_usd": cost_usd,
        "escalated": len(unresolved) if args.escalate else 0,
        "output_file": output_path,
    }

    if args.json_output:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"\n[{args.pub_id}] {summary['total']} refs in {elapsed_total:.1f}s")
        print(f"  VERIFIED: {summary['verified']}  MISMATCH: {summary['mismatch']}  "
              f"NOT_FOUND: {summary['not_found']}  UNVERIFIABLE: {summary['unverifiable']}")
        if args.escalate and unresolved:
            print(f"  Escalated: {len(unresolved)} refs to LLM")
        print(f"  Output: {output_path}")

    if summary["not_found"] > 0 or summary["mismatch"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
