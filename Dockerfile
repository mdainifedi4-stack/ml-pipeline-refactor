FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Create output directories
RUN mkdir -p outputs/plots outputs/models outputs/processed

# Set Python path
ENV PYTHONPATH=/app

# Default: run the pipeline test
CMD ["python", "tests/test_pipeline.py"]