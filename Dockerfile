# Dockerfile for Spatio-Temporal GNN & Active Learning Pipeline
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install pytest && \
    apt-get purge -y --auto-remove gcc g++

# Copy source code
COPY . .

# Expose port for potential web interface (optional)
EXPOSE 8000

# Default command
CMD ["python", "main.py", "--help"]
