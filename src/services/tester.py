import json
import os
import subprocess
import sys
from typing import Optional

from services.model import ModelService


class TesterService:

    def __init__(self, model: ModelService):
        self.model = model

    def generate(self, original_docs: list, output_dir: str) -> Optional[str]:
        """Generate a test file from *original_docs* and write it to *output_dir*.

        Returns the absolute path of the written test file, or None if generation
        failed.
        """
        print("[forge] Generating equivalence tests...")
        try:
            test_data = self.model.generate_test(
                json.dumps(original_docs, ensure_ascii=False, indent=2)
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"[forge] WARNING: Test generation failed: {exc}", file=sys.stderr)
            return None

        tokens = test_data.get("tokens", {})
        print(
            f"[forge] Test generation complete "
            f"({test_data.get('time', 'n/a')}, {tokens.get('total', '?')} tokens)."
        )

        raw = test_data["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()

        try:
            doc = json.loads(raw)
            if isinstance(doc, list):
                doc = doc[0]
            file_name = doc.get("path", "test_equivalence.py")
            file_path = os.path.join(output_dir, file_name)
            os.makedirs(output_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc["body"])
            print(f"[forge] Test file written: {file_path}")
            return file_path
        except (json.JSONDecodeError, KeyError) as exc:
            print(
                f"[forge] WARNING: Could not parse test response: {exc}",
                file=sys.stderr,
            )
            return None

    def run(self, test_file_path: str, source_dir: str, output_dir: str) -> bool:
        """Run *test_file_path* against both *source_dir* and *output_dir*.

        Returns True only if both runs pass.
        """
        runs = [
            ("input (original)", os.path.abspath(source_dir)),
            ("output (numba-annotated)", os.path.abspath(output_dir)),
        ]
        all_passed = True
        for label, module_dir in runs:
            print(f"[forge] Running tests against {label}...")
            env = os.environ.copy()
            env["MODULE_DIR"] = module_dir
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file_path, "-v", "--tb=short"],
                env=env,
                check=False,
            )
            if result.returncode != 0:
                all_passed = False
                print(
                    f"[forge] WARNING: Tests failed against {label}.",
                    file=sys.stderr,
                )
        if all_passed:
            print("[forge] All equivalence tests passed for both input and output.")
        return all_passed
