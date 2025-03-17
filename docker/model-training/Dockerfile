FROM python:3.10-slim

WORKDIR /app

# Copy model-training specific requirements
COPY docker/model-training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY setup.py .

# Create directories for models, features, data, and logs
RUN mkdir -p data/results data/features models logs

# Set environment variables
ENV PYTHONPATH=/app

# Install git
RUN apt-get update && apt-get install -y git

# Copy config files
COPY config/ /app/config/

# Copy and fix entrypoint script
COPY docker/model-training/entrypoint.sh /app/entrypoint.sh
RUN apt-get update && apt-get install -y --no-install-recommends dos2unix && \
    dos2unix entrypoint.sh && chmod +x entrypoint.sh

# Set entrypoint to run the script
ENTRYPOINT ["/app/entrypoint.sh"]
