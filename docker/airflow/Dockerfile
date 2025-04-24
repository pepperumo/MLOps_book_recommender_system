FROM apache/airflow:2.7.3-python3.9

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl \
    wget \
    rsync \
    docker.io \
    docker-compose \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies
COPY docker/airflow/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set work directory
WORKDIR /opt/airflow