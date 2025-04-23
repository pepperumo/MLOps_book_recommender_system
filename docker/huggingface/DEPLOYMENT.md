# Book Recommender System - Hugging Face Deployment Guide

This guide explains how to deploy your Book Recommender System full-stack application to Hugging Face Spaces.

## Prerequisites

Before deploying to Hugging Face Spaces, ensure you have:

- A Hugging Face account
- Your model file `collaborative.pkl` available locally
- Any required data files for your application

## Deployment Steps

### 1. Create a New Hugging Face Space

1. Log in to [Hugging Face](https://huggingface.co/)
2. Navigate to Spaces and click "Create new Space"
3. Give your Space a name (e.g., "book-recommender-system")
4. Select "Docker" as the SDK
5. Choose whether to make it public or private
6. Create the Space

### 2. Connect Your Repository

Connect your GitHub repository to the Hugging Face Space:

1. In your Space, go to the "Settings" tab
2. Under "Repository", select "Link external GitHub repository"
3. Enter your repository details
4. Set the Docker path to `docker/huggingface/Dockerfile`

### 3. Upload Your Model and Data Files

After the initial deployment (which will create placeholder files), you'll need to upload your actual model and data files:

1. Go to the "Files" tab in your Space
2. Navigate to `/app/models/`
3. Upload your `collaborative.pkl` file, replacing the placeholder
4. If needed, navigate to `/app/data/` and upload any required data files

### 4. Restart Your Space

After uploading your files:

1. Go to the "Settings" tab
2. Scroll down to "Factory reboot"
3. Click "Reboot" to restart your application with the new files

## Local Testing

To test the Hugging Face deployment configuration locally before deploying:

```bash
# Build and run the Hugging Face container
docker-compose -f docker-compose.huggingface.yml up --build
```

Then visit `http://localhost:7860` to access the application.

## Troubleshooting

### Application shows "Setup Required" page

This means the model file is missing. Follow the steps in the "Upload Your Model and Data Files" section.

### Port or connectivity issues

The application should be accessible on port 7860, which is the standard port for Hugging Face Spaces. If you're experiencing connectivity issues:

1. Check the Space logs for any error messages
2. Ensure the model file is properly uploaded and in the correct location
3. Verify that the FastAPI server is running inside the container

## Architecture

This deployment uses a multi-container approach:

- A React frontend served by Nginx
- A FastAPI backend
- A Python proxy to handle routing between services and expose port 7860

All components run in a single Docker container, making it simple to deploy to Hugging Face Spaces.

## Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/getting-started.html)