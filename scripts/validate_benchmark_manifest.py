"""Validate the merged benchmark manifest against the local paper corpus."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - environment error
    raise SystemExit("PyMuPDF is required: install package 'PyMuPDF'.") from exc


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmark" / "benchmark_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    papers = manifest.get("papers")
    if not isinstance(papers, list) or len(papers) != 13:
        fail(f"expected exactly 13 papers, got {len(papers) if isinstance(papers, list) else type(papers)}")

    profiles = manifest["scheme_profiles"]
    ids = [paper["paper_id"] for paper in papers]
    if len(ids) != len(set(ids)):
        fail("paper_id values are not unique")

    for paper in papers:
        paper_id = paper["paper_id"]
        path = ROOT / paper["file"]
        if not path.is_file():
            fail(f"{paper_id}: missing file {path}")
        actual_hash = sha256(path)
        if actual_hash != paper["sha256"]:
            fail(f"{paper_id}: SHA-256 mismatch: {actual_hash} != {paper['sha256']}")

        with fitz.open(path) as document:
            actual_pages = document.page_count
        if actual_pages != paper["page_count"]:
            fail(f"{paper_id}: page count mismatch: {actual_pages} != {paper['page_count']}")

        scheme = paper["scheme"]
        if scheme not in profiles:
            fail(f"{paper_id}: unknown scheme {scheme!r}")
        expected = profiles[scheme]
        if paper["parameters"] != expected:
            fail(f"{paper_id}: parameters do not match {scheme} profile")

        factor = paper["reduction"]["correction_factor"]
        if not isinstance(factor, int) or isinstance(factor, bool):
            fail(f"{paper_id}: correction_factor must be an integer")

        for evidence in paper.get("evidence", []):
            for page in evidence.get("pages", []):
                if not isinstance(page, int) or not 1 <= page <= actual_pages:
                    fail(f"{paper_id}: evidence page {page!r} outside 1..{actual_pages}")

        if paper.get("annotation_status") != "annotated":
            fail(f"{paper_id}: annotation_status is not annotated")

    print(f"validated {len(papers)} papers")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, json.JSONDecodeError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
