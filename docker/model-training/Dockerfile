FROM python:3.10-slim

WORKDIR /app

# Copy Docker-specific requirements
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .

# Create directories for data, models, and logs
RUN mkdir -p data/processed data/features models logs

# Set environment variables
ENV PYTHONPATH=/app

# Create an entry point script
COPY docker/model-training/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Train the models
ENTRYPOINT ["./entrypoint.sh"]
