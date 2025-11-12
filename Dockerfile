# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app ./app

# Expose backend port (default 8000, can be overridden via environment variable)
EXPOSE 8000

# Run the backend with configurable host and port
# Environment variables can be set at runtime via docker-compose or Kubernetes
CMD uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}