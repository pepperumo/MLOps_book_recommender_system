# MLOps Book Recommender System

## Deployment Options

### Option 1: Full Pipeline (Data Processing, Model Training, API, Frontend)
This option runs the complete pipeline from data retrieval to model training and deployment:

```bash
docker-compose up
```

After the pipeline completes and models are generated, you can deploy the API and frontend:

```bash
docker-compose -f docker-compose.deploy.yml up
```

### Option 2: Deployment Only (API and Frontend)
Use this option when you already have the necessary data and trained models:

```bash
docker-compose -f docker-compose.deploy.yml up
```

**Requirements:**
- The `./models/collaborative.pkl` model file must exist
- The `./data/processed/` directory must contain the processed data files

### Option 3: Monitoring Services
You can run monitoring services in three different ways:

#### Standalone Monitoring (no API access)
```bash
docker-compose -f docker-compose.monitoring.yml up
```

#### With Deployment (for production use)
```bash
# Start deployment services first
docker-compose -f docker-compose.deploy.yml up -d

# Then start monitoring services
docker-compose -f docker-compose.monitoring.yml up
```

#### All-In-One Combined (recommended for production)
```bash
docker-compose -f docker-compose.deploy.yml -f docker-compose.monitoring.yml up
```

This combined approach creates a shared network automatically, allowing all services to communicate.

## Troubleshooting

### FastAPI Service Fails to Start
If the FastAPI service fails with an error about missing models:

1. Ensure the `./models/collaborative.pkl` file exists 
2. Run the full pipeline first: `docker-compose up`
3. Or manually place your trained model file in the `./models/` directory

### Frontend Cannot Connect to API
If the frontend loads but cannot connect to the API:

1. Check that the FastAPI service is running and healthy
2. Verify that port 9998 is accessible
3. Check browser console for CORS or connection errors

### Prometheus Cannot Connect to FastAPI
If Prometheus shows connection errors to the FastAPI service:

1. Make sure both services are on the same Docker network
2. When running monitoring standalone, this is expected
3. Adjust the prometheus.yml file if needed to match your network configuration