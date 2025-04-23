FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git openssh-client dos2unix curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY docker/dvc/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Entrypoint
COPY docker/dvc/entrypoint.sh /entrypoint.sh
RUN dos2unix /entrypoint.sh && chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["/entrypoint.sh"]