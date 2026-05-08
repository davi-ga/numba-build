import json
import os
from typing import List, Dict, Optional


class PatcherService:

    def discover(
        self,
        root: str,
        extensions: Optional[List[str]] = None,
        ignore_dirs: Optional[List[str]] = None,
    ) -> List[str]:

        ignore_dirs = set(
            ignore_dirs or [".git", "__pycache__", ".venv", "node_modules"]
        )
        relative_paths = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

            for filename in filenames:
                if extensions and not any(filename.endswith(ext) for ext in extensions):
                    continue
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, root)
                relative_paths.append(rel_path)

        return sorted(relative_paths)

    def _pathing(self, root: str, relative_paths: List[str]) -> List[Dict[str, str]]:
        documents = []
        for rel_path in relative_paths:
            abs_path = os.path.join(root, rel_path)
            with open(abs_path, "r", encoding="utf-8") as f:
                documents.append({"path": rel_path, "body": f.read()})
        return documents

    def to_json(self, root: str, relative_paths: List[str]) -> str:
        documents = self._pathing(root, relative_paths)
        return json.dumps(documents, ensure_ascii=False, indent=2)

    def to_files(self, output_dir: str, raw_json: str) -> List[str]:
        documents: List[Dict[str, str]] = (
            json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        )

        created = []
        for doc in documents:
            rel_path = doc["path"]
            body = doc["body"]

            abs_output = os.path.realpath(output_dir)
            abs_path = os.path.realpath(os.path.join(output_dir, rel_path))
            if not abs_path.startswith(abs_output + os.sep) and abs_path != abs_output:
                raise ValueError(
                    f"Refusing to write outside output directory: '{rel_path}'"
                )
            os.makedirs(os.path.dirname(abs_path) or output_dir, exist_ok=True)

            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(body)

            created.append(abs_path)

        return created
