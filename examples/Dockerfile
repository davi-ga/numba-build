FROM python:3.12-slim

# libgomp1 is required by numba at runtime (OpenMP threading)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Web server entrypoint
COPY app.py .

# Numba-optimised helpers produced by numba-build during the CI/CD pipeline.
# These replace utils/ at runtime with @numba.njit-annotated versions.
COPY optimized/ optimized/

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]
