"""
numba-build — LLM-based build-time refactoring pipeline.

Discovers Python source files, sends them to the Gemini LLM for
modularisation, injects @numba.njit decorators via AST, and writes the
optimised artefacts to an output directory ready to be copied into a
Cloud Run container.

CLI usage (after pip install numba-runtime-middleware[build]):
    numba-build --source-dir my_project/ --output-dir dist/

Environment variables:
    SOURCE_DIR      Override --source-dir default
    OUTPUT_DIR      Override --output-dir default
    GEMINI_API_KEY  Gemini API key (required)
    PROMPT          System prompt sent to the LLM (required)
"""

import argparse
import json
import os
import sys

from services.annotator import AnnotatorService
from services.model import ModelService
from services.patcher import PatcherService
import py_compile, tempfile, os


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="numba-build",
        description=(
            "LLM-based code refactoring step for CI/CD pipelines. "
            "Transforms plain Python source files into Numba-optimised "
            "artefacts ready for Cloud Run deployment."
        ),
    )
    parser.add_argument(
        "--source-dir",
        default=os.getenv("SOURCE_DIR", "sample_project"),
        help="Directory containing the source Python files (default: sample_project).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "output_dir"),
        help="Directory where the refactored files will be written (default: output_dir).",
    )
    return parser.parse_args()


def _validate_env() -> None:
    missing = [v for v in ("GEMINI_API_KEY", "PROMPT") if not os.getenv(v)]
    if missing:
        print(
            f"[numba-build] ERROR: Missing required environment variable(s): "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _annotate_documents(
    annotator: AnnotatorService,
    documents: list[dict],
) -> list[dict]:
    annotated = []
    for doc in documents:
        try:
            annotated_body = annotator.transform(doc["body"])
        except SyntaxError as exc:
            print(
                f"[numba-build] WARNING: Skipping annotation for '{doc['path']}' "
                f"(SyntaxError): {exc}",
                file=sys.stderr,
            )
            annotated_body = doc["body"]
        annotated.append({"path": doc["path"], "body": annotated_body})
    return annotated


def _validate_documents(documents: list) -> None:
    """Raise ValueError if any document is missing required keys."""
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(f"Item {i} is not an object, got {type(doc).__name__}.")
        missing = [k for k in ("path", "body") if k not in doc]
        if missing:
            raise ValueError(
                f"Item {i} is missing required key(s): {', '.join(missing)}."
            )


def run(source_dir: str, output_dir: str) -> list[str]:
    """
    Execute the full build pipeline programmatically.

    Parameters
    ----------
    source_dir:
        Directory containing the raw Python source files to optimise.
    output_dir:
        Directory where the refactored, Numba-annotated files will be written.

    Returns
    -------
    List of absolute paths of the files written to output_dir.

    Raises
    ------
    EnvironmentError — GEMINI_API_KEY or PROMPT not set.
    RuntimeError     — LLM call failed or returned invalid output.
    """
    missing = [v for v in ("GEMINI_API_KEY", "PROMPT") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    patcher = PatcherService()
    model = ModelService()
    annotator = AnnotatorService()

    # Step 1 — Discover
    print(f"[numba-build] Discovering Python files in '{source_dir}'...")
    paths = patcher.discover(source_dir, extensions=[".py"])
    if not paths:
        raise RuntimeError(f"No Python files found in '{source_dir}'.")
    print(f"[numba-build] Found {len(paths)} file(s): {paths}")

    # Step 2 — LLM inference
    payload = patcher.to_json(source_dir, paths)
    print("[numba-build] Calling LLM for modularisation...")
    try:
        model_data = model.modularize(payload)
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    tokens = model_data.get("tokens", {})
    print(
        f"[numba-build] LLM response received "
        f"({model_data.get('time', 'n/a')}, {tokens.get('total', '?')} tokens)."
    )

    # Step 3 — Parse LLM output
    raw_text: str = model_data["text"]
    
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]  # drop opening fence line
        stripped = stripped.rsplit("```", 1)[0]  # drop closing fence
        stripped = stripped.strip()
    try:
        documents: list[dict] = json.loads(stripped)
        if not isinstance(documents, list):
            raise ValueError("LLM output is not a JSON array.")
        _validate_documents(documents)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"LLM did not return a valid JSON array of files: {exc}\n"
            f"Raw output (first 500 chars): {raw_text}"
        ) from exc

    for doc in documents:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(doc["body"].encode())
            tmp = f.name
        try:
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as exc:
            raise exc
        finally:
            os.unlink(tmp)

    # Step 4 — Numba annotation
    print(f"[numba-build] Annotating {len(documents)} file(s) with @numba.njit...")
    annotated_documents = _annotate_documents(annotator, documents)

    # Step 5 — Write artefacts
    print(f"[numba-build] Writing optimised files to '{output_dir}'...")
    created = patcher.to_files(output_dir, annotated_documents)

    print(f"[numba-build] Done. {len(created)} file(s) written:")
    for path in created:
        print(f"  - {path}")

    return created


def main() -> None:
    """Entry point for the `numba-build` CLI command."""
    args = _parse_args()
    _validate_env()
    try:
        run(args.source_dir, args.output_dir)
    except (EnvironmentError, RuntimeError) as exc:
        print(f"[numba-build] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
