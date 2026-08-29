# Dockerfile for Abuse-Ring Sentinel Backend API
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/

# Expose ports
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run uvicorn
CMD ["python", "-m", "uvicorn", "backend.sentinel_app:app", "--host", "0.0.0.0", "--port", "8000"]
