FROM python:3.10-slim

WORKDIR /app

# Copy Docker-specific requirements
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .

# Create directories for data and logs
RUN mkdir -p data/raw logs

# Set environment variables
ENV PYTHONPATH=/app

# Create an entry point script
COPY docker/data-retrieval/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Run the data retrieval process
ENTRYPOINT ["./entrypoint.sh"]
