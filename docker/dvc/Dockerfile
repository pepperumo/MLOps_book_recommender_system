FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    dos2unix \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY docker/dvc/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy entrypoint script
COPY docker/dvc/entrypoint.sh /entrypoint.sh
RUN dos2unix /entrypoint.sh && chmod +x /entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Define the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
