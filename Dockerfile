# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install curl
RUN apt update && apt install -y curl

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files into container
COPY . .
COPY ./mlruns /app/mlruns
COPY scaler.pkl /app/

# Expose FastAPI port
EXPOSE 8000

# Set PYTHONPATH to include /app (so 'src' becomes importable)
ENV PYTHONPATH="/app"

# Run FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
