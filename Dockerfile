FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / torch (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api.py .

# Outputs directory (review JSONs are optional; the API returns them in-memory)
RUN mkdir -p outputs

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
