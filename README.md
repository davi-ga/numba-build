# numba-build

LLM as an Intelligent Compiler — uses the Gemini API at **build time** to modularise Python source code and inject `@numba.njit` decorators, then provides a lightweight **runtime middleware** for Cloud Run containers with warm-up, telemetry and fallback.

---

## How it works

```
BUILD TIME (CI/CD)                         RUNTIME (Cloud Run)
──────────────────────────────────         ──────────────────────────────────
numba-build --source-dir src/              Middleware(modules=[...])
  │                                          │
  ├── PatcherService                         ├── warmup()
  │   Discovers .py files                    │   Triggers Numba JIT compilation
  │   Serialises to JSON payload             │   before the first request
  │                                          │
  ├── ModelService (Gemini API)              ├── call("func_name", *args)
  │   Sends code to the LLM                  │   Executes with telemetry
  │   Receives modularised code back         │   CPU · RAM · wall-clock time
  │                                          │
  ├── AnnotatorService (AST)                 └── Fallback
  │   Injects import numba                       If @numba.njit raises,
  │   Adds @numba.njit to eligible functions     re-runs with original callable
  │
  └── output_dir/
      Optimised artefacts → Docker image
```

---

## Installation

**Runtime only** (Cloud Run container — minimal dependencies):
```bash
pip install git+https://github.com/davi-ga/numba-build
```

**Build + Runtime** (CI/CD pipeline — includes Gemini SDK):
```bash
pip install "git+https://github.com/davi-ga/numba-build[build]"
```

---

## Build pipeline

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini API key |
| `PROMPT` | Yes | System prompt instructing the LLM to return `[{path, body}]` JSON |
| `SOURCE_DIR` | No | Override `--source-dir` default |
| `OUTPUT_DIR` | No | Override `--output-dir` default |

### CLI

```bash
numba-build --source-dir my_project/ --output-dir dist/
```

### Python API

```python
from build_pipeline import run

created_files = run(source_dir="my_project/", output_dir="dist/")
```

### Cloud Build (`cloudbuild.yaml`)

```yaml
steps:
  - name: "python:3.12-slim"
    entrypoint: pip
    args: ["install", "--no-cache-dir", "numba-runtime-middleware[build]@git+https://github.com/davi-ga/numba-build.git"]

  - name: "python:3.12-slim"
    entrypoint: numba-build
    args: ["--source-dir", "my_project/", "--output-dir", "dist/"]
    secretEnv: ["GEMINI_API_KEY", "PROMPT"]

  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "-t", "gcr.io/$PROJECT_ID/my-service", "."]
```

---

## Runtime middleware

```python
import calculator  # module from output_dir/, already in WORKDIR
from middleware import Middleware

# Register modules and optional fallbacks for each function
mw = Middleware(
    modules=[calculator],
    fallbacks={"divide": original_divide},  # optional
)

# Trigger Numba JIT compilation on container startup (warm-up)
warmup_report = mw.warmup()
# {"add": {"status": "ok", "compile_ms": 312.4}, ...}

# Execute with full telemetry
result = mw.call("add", 1, 2)
# {
#   "result": 3,
#   "time_ms": 0.04,
#   "cpu_avg": 0.1,
#   "cpu_max": 0.3,
#   "mem_mb_avg": 48.2,
#   "mem_mb_max": 48.5,
#   "fallback_used": False
# }
```

### `warmup(inputs=None)`

| Parameter | Type | Description |
|---|---|---|
| `inputs` | `dict[str, tuple] \| None` | `{func_name: (args,...)}` for domain-accurate warm-up inputs. Functions without an entry receive `np.zeros(1, dtype=np.float64)` as default. |

### `call(func_name, *args, **kwargs)`

Executes the optimised function. If it raises and a fallback is registered, runs the fallback instead and sets `"fallback_used": True` in the response.

Raises `KeyError` if the function name is not registered.  
Raises `RuntimeError` if the function raises and no fallback is available.

---

## Dockerfile (Cloud Run)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the LLM-generated, Numba-annotated artefacts enter the image.
# The raw source directory is intentionally excluded.
COPY dist/ .

EXPOSE 8080
CMD ["python", "main.py"]
```

---

## Project structure

```
src/
  build_pipeline.py   # Build-time pipeline — CLI + Python API
  middleware.py       # Runtime middleware — warm-up, telemetry, fallback
  services/
    model.py          # Gemini API client
    annotator.py      # AST transformer (@numba.njit injection)
    patcher.py        # File discovery and I/O
  utils/
    modifier.py       # ast.NodeTransformer (Inserter)
refactor_script.py    # Local convenience wrapper for development
cloudbuild.yaml       # GCP Cloud Build pipeline
Dockerfile.optimized  # Production container image
pyproject.toml        # Package metadata and CLI registration
```

---

## Requirements

- Python ≥ 3.11
- Runtime: `numpy`, `psutil`, `numba`
- Build extras: `google-genai`, `python-dotenv`
