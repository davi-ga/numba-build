"""
forge — LLM-based build-time refactoring pipeline.

Discovers Python source files, sends them to the Gemini LLM for
modularisation, injects @numba.njit decorators via AST, and writes the
optimised artefacts to an output directory ready to be copied into a
Cloud Run container.

CLI usage (after pip install sentinel[build]):
    forge --source-dir my_project/ --output-dir dist/

Environment variables:
    SOURCE_DIR      Override --source-dir default
    OUTPUT_DIR      Override --output-dir default
    GEMINI_API_KEY  Gemini API key (required)
    MODULARIZE_PROMPT   System prompt sent to the LLM (required)
    TEST_PROMPT         Test prompot sent to the LLM (required)
    
"""

import argparse
import json
import os
import py_compile
import sys
import tempfile

from services.annotator import AnnotatorService
from services.compatibility import NumbaCompatibilityChecker
from services.model import ModelService
from services.patcher import PatcherService
from services.preprocessor import PreprocessorService
from services.tester import TesterService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="forge",
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
    missing = [
        v
        for v in ("GEMINI_API_KEY", "MODULARIZE_PROMPT", "TEST_PROMPT")
        if not os.getenv(v)
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )


def _annotate_documents(
    annotator: AnnotatorService,
    documents: list[dict],
) -> list[dict]:
    annotated = []
    for doc in documents:
        path = doc["path"]
        body = doc["body"]

        if "__init__" in path:
            annotated.append({"path": path, "body": body})
            continue

        try:
            body = annotator.transform(body)
        except SyntaxError as exc:
            print(
                f"[forge] WARNING: Skipping annotation for '{path}' "
                f"(SyntaxError): {exc}",
                file=sys.stderr,
            )

        annotated.append({"path": path, "body": body})
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


def run(
    source_dir: str,
    output_dir: str,
    max_attempts: int | None = None,
) -> list[str]:
    """
    Execute the full build pipeline programmatically.

    Parameters
    ----------
    source_dir:
        Directory containing the raw Python source files to optimise.
    output_dir:
        Directory where the refactored, Numba-annotated files will be written.
    max_attempts:
        Maximum LLM retry attempts when equivalence tests fail.
        Defaults to FORGE_MAX_ATTEMPTS env var, or 3.

    Returns
    -------
    List of absolute paths of the files written to output_dir.

    Raises
    ------
    EnvironmentError — GEMINI_API_KEY, MODULARIZE_PROMPT or TEST_PROMPT not set.
    RuntimeError     — LLM call failed or returned invalid output.
    """

    _validate_env()
    if max_attempts is None:
        max_attempts = int(os.getenv("FORGE_MAX_ATTEMPTS", "3"))

    patcher = PatcherService()
    model = ModelService()
    annotator = AnnotatorService()
    tester = TesterService(model)
    preprocessor = PreprocessorService()
    compat_checker = NumbaCompatibilityChecker()

    # Step 1 — Discover
    print(f"[forge] Discovering Python files in '{source_dir}'...")
    paths = patcher.discover(source_dir, extensions=[".py"])
    if not paths:
        raise RuntimeError(f"No Python files found in '{source_dir}'.")
    print(f"[forge] Found {len(paths)} file(s): {paths}")

    payload = patcher.to_json(source_dir, paths)
    original_docs = json.loads(payload)

    # Step 1.5 — Preprocess (AST transformers before LLM)
    print(f"[forge] Preprocessing {len(original_docs)} file(s)...")
    preprocessed_docs = preprocessor.transform_documents(original_docs)
    payload = json.dumps(preprocessed_docs, ensure_ascii=False, indent=2)

    # Generate equivalence tests once — re-used across every LLM retry
    test_file_path = tester.generate(original_docs, output_dir)

    prev_created: list[str] = []
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(
                f"[forge] Retrying LLM pipeline (attempt {attempt}/{max_attempts})..."
            )
            for path in prev_created:
                try:
                    os.remove(path)
                except OSError:
                    pass

        # Step 2 — LLM inference
        print("[forge] Calling LLM for modularization...")
        try:
            model_data = model.modularize(payload)
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        tokens = model_data.get("tokens", {})
        print(
            f"[forge] LLM response received "
            f"({model_data.get('time', 'n/a')}, {tokens.get('total', '?')} tokens)."
        )

        # Step 3 — Parse LLM output
        raw_text: str = model_data["text"]

        stripped = raw_text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]
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
                raise RuntimeError(f"Syntax error in '{doc['path']}': {exc}") from exc
            finally:
                os.unlink(tmp)

        # Step 3.5 — Numba compatibility check
        print("[forge] Checking Numba compatibility...")
        all_compatible = True
        for doc in documents:
            is_compatible, issues = compat_checker.check(doc["body"])
            if not is_compatible:
                all_compatible = False
                print(
                    f"[forge] WARNING: Compatibility issues in '{doc['path']}':",
                    file=sys.stderr,
                )
                for severity, line, message in issues:
                    print(f"  [{severity.upper()}] Line {line}: {message}", file=sys.stderr)

        if not all_compatible:
            if attempt < max_attempts:
                print(
                    f"[forge] WARNING: Compatibility issues found on attempt {attempt}. "
                    "Reprocessing with LLM...",
                    file=sys.stderr,
                )
                continue
            else:
                print(
                    f"[forge] WARNING: Compatibility issues remain after {max_attempts} attempt(s). "
                    "Proceeding with annotation (some functions may fail at runtime).",
                    file=sys.stderr,
                )

        # Step 4 — Numba annotation
        print(f"[forge] Annotating {len(documents)} file(s) with @numba.njit...")
        annotated_documents = _annotate_documents(annotator, documents)

        # Step 5 — Write artefacts
        print(f"[forge] Writing optimised files to '{output_dir}'...")
        created = patcher.to_files(output_dir, annotated_documents)
        helpers_path = patcher.emit_helpers_module(output_dir)
        created.append(helpers_path)
        prev_created = list(created)

        # Step 6 — Run equivalence tests
        if test_file_path:
            if test_file_path not in created:
                created.append(test_file_path)
            tests_passed = tester.run(test_file_path, source_dir, output_dir)
            if not tests_passed:
                if attempt < max_attempts:
                    print(
                        f"[forge] WARNING: Tests failed on attempt {attempt}. "
                        "Reprocessing with LLM...",
                        file=sys.stderr,
                    )
                    continue
                else:
                    raise RuntimeError(
                        f"Equivalence tests failed after {max_attempts} attempt(s). "
                        "Aborting."
                    )
        break

    print(f"[forge] Done. {len(created)} file(s) written:")
    for path in created:
        print(f"  - {path}")

    return created


def main() -> None:
    """Entry point for the `forge` CLI command."""
    args = _parse_args()
    try:
        run(args.source_dir, args.output_dir)
    except (EnvironmentError, RuntimeError) as exc:
        print(f"[forge] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
