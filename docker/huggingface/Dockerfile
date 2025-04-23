# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy all frontend files
COPY frontend/ ./

# Build the application
RUN npm run build && ls -la build

# Stage 2: Set up Python and FastAPI
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy FastAPI-specific requirements
COPY docker/huggingface/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY setup.py .

# Create directories for model, data, and logs
RUN mkdir -p /app/models /app/data/processed /app/logs



# Copy the built frontend from the first stage
COPY --from=frontend-build /app/build /usr/share/nginx/html

# Verify frontend files
RUN ls -la /usr/share/nginx/html

# Copy Nginx configuration - using our Hugging Face specific config
COPY docker/huggingface/nginx.conf /etc/nginx/conf.d/default.conf
# Remove the default nginx welcome page config
RUN rm -f /etc/nginx/sites-enabled/default /etc/nginx/sites-available/default

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV MODEL_PATH=/app/models/collaborative.pkl

# Prepare the entrypoint script
COPY docker/huggingface/entrypoint.sh /app/entrypoint.sh
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose port 7860 (standard for Hugging Face Spaces)
EXPOSE 7860

# Set entrypoint to run both services
ENTRYPOINT ["/app/entrypoint.sh"]