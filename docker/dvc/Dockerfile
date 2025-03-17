FROM python:3.10-slim

# Install system dependencies needed for DVC and Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client

# Set working directory explicitly
WORKDIR /app

# Copy DVC-specific requirements and install dependencies
COPY docker/dvc/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (including DVC files, if present)
COPY . /app/

# Ensure required directories exist
RUN mkdir -p /app/logs /app/models /app/data

# Copy a default config folder if needed:
COPY config/ /app/config/

# Copy and fix the entrypoint script
COPY docker/dvc/entrypoint.sh /app/entrypoint.sh
RUN apt-get update && apt-get install -y --no-install-recommends dos2unix && \
    dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set environment variables (optional, but useful)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Define the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
