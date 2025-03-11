FROM python:3.10-slim

WORKDIR /app

# Setting environment variables to avoid binary incompatibility issues
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies required for numpy/scipy/scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies in a specific order
# to ensure binary compatibility
COPY src/api/requirements.txt .

# Install dependencies with specific order to ensure compatibility
RUN pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir scipy==1.10.1 && \
    pip install --no-cache-dir pandas==1.5.3 && \
    pip install --no-cache-dir scikit-learn==1.2.2 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the API code
COPY src/api /app/src/api/
COPY src/models /app/src/models/
COPY src/data /app/src/data/

# Create necessary directories
RUN mkdir -p /app/models
RUN mkdir -p /app/logs
RUN mkdir -p /app/data/processed

# Create a non-root user and switch to it
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with our enhanced entry point script
CMD ["python", "src/api/docker_entrypoint.py"]
