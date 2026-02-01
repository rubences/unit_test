# Moto-Edge-RL Docker Image

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p data/raw data/processed models logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MOTO_EDGE_RL_HOME=/app

# Default command
CMD ["python", "-m", "moto_edge_rl.train"]
