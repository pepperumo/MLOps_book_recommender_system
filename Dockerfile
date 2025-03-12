FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Make sure the data and models directories exist
RUN mkdir -p data/raw data/processed data/features data/results models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
