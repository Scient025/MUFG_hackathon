FROM python:3.11-slim

WORKDIR /app

# Install only required system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libasound2 \
    libasound2-dev \
    libgomp1 \
    libcurl4 \
    ca-certificates \
    libuuid1 \
    libkrb5-3 \
    zlib1g \
    libglib2.0-0 \
    libnss3 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check (using Python instead of curl)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
