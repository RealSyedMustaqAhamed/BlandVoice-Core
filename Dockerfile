# Production Dockerfile for BlandVoice-Core Audio Research Prototype
# Optimized for Hugging Face Spaces, Render, Railway, or Google Cloud Run

FROM python:3.11-slim

# Install system audio dependencies (libsndfile & ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Ensure non-buffered stdout for real-time streaming logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install lightweight PyTorch CPU first to leverage Docker layer caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create a non-root user (required for Hugging Face Spaces security policy)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose default application port
EXPOSE 7860

# Launch Uvicorn ASGI server with dynamic port support
CMD ["sh", "-c", "uvicorn demo.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
