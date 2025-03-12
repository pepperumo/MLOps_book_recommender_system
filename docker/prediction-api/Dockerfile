FROM python:3.10-slim

WORKDIR /app

# Copy Docker-specific requirements and add FastAPI dependencies
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Copy source code
COPY src/ ./src/
COPY setup.py .

# Create directories for data, models, and logs
RUN mkdir -p data/processed models logs

# Set environment variables
ENV PYTHONPATH=/app

# Copy the API code
COPY docker/prediction-api/api.py .
COPY docker/prediction-api/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose the API port
EXPOSE 8000

# Start the API server
ENTRYPOINT ["./entrypoint.sh"]
