# Book Recommender System - Hugging Face Deployment

This is a full-stack book recommender system application, consisting of a FastAPI backend and a React frontend, packaged for deployment on Hugging Face Spaces.

## About this Application

This application provides book recommendations using a collaborative filtering model. The frontend allows users to browse books and get personalized recommendations, while the backend serves the recommendations through an API.

## Running Locally

To test this deployment locally before pushing to Hugging Face:

```bash
# Clone the repository
git clone <your-repo-url>
cd MLOps_book_recommender_system

# Make sure you have the model and data files in the right locations
# - models/collaborative.pkl
# - data/ (with necessary datasets)

# Run using docker-compose
docker-compose -f docker-compose.huggingface.yml up
```

Visit http://localhost:7860 to access the application.

## Deploying to Hugging Face Spaces

To deploy this application on Hugging Face Spaces:

1. Make sure your repository includes all necessary files:
   - The model file at `models/collaborative.pkl`
   - Required data files in the `data/` directory
   - All source code and configuration files

2. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" as the Space SDK
   - Connect your GitHub repository

3. Configure the Space:
   - Set the Dockerfile path to `docker/huggingface/Dockerfile`
   - Set appropriate hardware (recommended: CPU with at least 4GB RAM)
   - Enable "Always on" if you want the application to be continuously available

4. Wait for the build to complete and your application will be live!

## Structure

- `/docker/huggingface/Dockerfile`: Multi-stage Docker build that creates both the frontend and backend
- `/docker/huggingface/entrypoint.sh`: Script that starts all services and sets up a proxy for Hugging Face
- `/docker-compose.huggingface.yml`: Docker Compose configuration for testing locally

## API Documentation

Once deployed, API documentation is available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Credits

This application is part of the MLOps Book Recommender System project.