# Book Recommender System API

This API provides access to the book recommender system, allowing you to retrieve book information, get personalized recommendations, and more.

## Features

- **Book Information**: Browse and search for books with filtering options
- **Personalized Recommendations**: Get book recommendations for specific users
- **Similar Books**: Find books similar to a book of interest
- **Rating Submission**: Add new user ratings (requires API key)

## API Endpoints

### Books

- `GET /books`: Get a paginated list of books with optional filtering
  - Query parameters:
    - `skip`: Number of books to skip (default: 0)
    - `limit`: Number of books to return (default: 10, max: 100)
    - `min_rating`: Minimum average rating (1-5)
    - `author`: Filter by author name (case-insensitive partial match)

- `GET /books/{book_id}`: Get detailed information for a specific book

### Recommendations

- `GET /recommendations/user/{user_id}`: Get book recommendations for a specific user
  - Query parameters:
    - `num`: Number of recommendations to return (default: 5, max: 20)
    - `strategy`: Recommendation strategy (collaborative, content, or hybrid)

- `GET /recommendations/similar-to/{book_id}`: Get books similar to a specified book
  - Query parameters:
    - `num`: Number of similar books to return (default: 5, max: 20)

### Ratings

- `POST /ratings`: Submit a new user rating for a book (requires API key)
  - Request body:
    ```json
    {
      "user_id": 123,
      "book_id": 456,
      "rating": 4.5
    }
    ```

### System

- `GET /health`: Check if the API is running

## Authentication

Some endpoints (like adding ratings) require authentication. Send your API key in the `X-API-Key` header:

```
X-API-Key: your-api-key
```

## Testing the API

Before deploying the API, you should test all its endpoints to ensure they're working properly:

1. Start the API server (if not already running):
   ```
   cd src/api
   uvicorn app:app --host 0.0.0.0 --port 8003
   ```

2. Run the automated test script to check all endpoints:
   ```
   cd <project_root>
   python src/api/test_api.py
   ```

   The test script checks all API endpoints with both valid and invalid inputs to ensure proper error handling.

3. To test against a different server URL (like a containerized version):
   ```
   python src/api/test_api.py --server-url http://localhost:8000
   ```

## Testing the Dockerized API

Once you have your API running in a Docker container, you can test it to verify it's working correctly and making good predictions. Here's a detailed guide:

### 1. Start the Containerized API

```bash
# Build and start the container
docker-compose up --build -d

# Verify the container is running
docker-compose ps

# Check logs to ensure proper startup
docker-compose logs
```

Look for messages indicating successful model loading and API startup.

### 2. Run the Test Script

Our test script automates checking all endpoints:

```bash
python src/api/test_api.py --server-url http://localhost:8000
```

### 3. Manual Testing Examples

You can also manually test specific endpoints with curl:

#### Health Check
```bash
curl http://localhost:8000/health
```
Expected output:
```json
{
  "status": "healthy",
  "timestamp": "2025-03-11T04:12:26.608327",
  "model_loaded": true
}
```

#### Get Book Details
```bash
curl http://localhost:8000/books/1
```
Expected output:
```json
{
  "book_id": 1,
  "title": "Harry Potter and the Half-Blood Prince (Harry Potter, #6)",
  "authors": "J.K. Rowling, Mary GrandPré",
  "average_rating": 4.54,
  "isbn": "439785960",
  "language_code": "eng",
  "original_publication_year": 2005.0
}
```

#### Get User Recommendations
```bash
curl http://localhost:8000/recommendations/user/1?limit=3
```
Expected output:
```json
[
  {
    "book_id": 28,
    "title": "Notes from a Small Island",
    "authors": "Bill Bryson",
    "average_rating": 3.91,
    "isbn": "380727501",
    "language_code": "eng",
    "original_publication_year": 1995.0
  },
  {
    "book_id": 9791,
    "title": "Angela's Ashes (Frank McCourt, #1)",
    "authors": "Frank McCourt",
    "average_rating": 4.09,
    "isbn": "7205599X",
    "language_code": "eng",
    "original_publication_year": 1996.0
  },
  {
    "book_id": 3,
    "title": "Harry Potter and the Sorcerer's Stone (Harry Potter, #1)",
    "authors": "J.K. Rowling, Mary GrandPré",
    "average_rating": 4.45,
    "isbn": "439554934",
    "language_code": "eng",
    "original_publication_year": 1997.0
  }
]
```

#### Get Similar Books
```bash
curl http://localhost:8000/recommendations/similar-to/1?limit=3
```
Expected output:
```json
[
  {
    "book_id": 2,
    "title": "Harry Potter and the Order of the Phoenix (Harry Potter, #5)",
    "authors": "J.K. Rowling, Mary GrandPré",
    "average_rating": 4.46,
    "score": 1.0
  },
  {
    "book_id": 6,
    "title": "Harry Potter and the Goblet of Fire (Harry Potter, #4)",
    "authors": "J.K. Rowling, Mary GrandPré",
    "average_rating": 4.53,
    "score": 0.75
  },
  {
    "book_id": 5,
    "title": "Harry Potter and the Prisoner of Azkaban (Harry Potter, #3)",
    "authors": "J.K. Rowling, Mary GrandPré",
    "average_rating": 4.54,
    "score": 0.5
  }
]
```

#### Test Error Handling
```bash
# Testing non-existent book
curl http://localhost:8000/books/999999
```
Expected output:
```json
{"detail":"Book with ID 999999 not found"}
```

```bash
# Testing non-existent user
curl http://localhost:8000/recommendations/user/999999
```
Expected output:
```json
{"detail":"User with ID 999999 not found"}
```

### 4. Submit Ratings (requires API key)

To test the rating submission (which requires authentication):

```bash
curl -X POST http://localhost:8000/ratings \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{"user_id": 1, "book_id": 1, "rating": 5.0}'
```

Expected output:
```json
{"status":"success","message":"Rating saved"}
```

### 5. Evaluating Prediction Quality

To evaluate if the system makes good predictions:

1. **Similar Books Test**: For book ID 1 (Harry Potter and the Half-Blood Prince), you should get other Harry Potter books as similar recommendations.

2. **User Recommendations Test**: Check if recommendations for a specific user align with books they might like based on:
   - Genre preferences (if the same genres appear in recommendations)
   - Author consistency (if authors they've liked before appear in recommendations)
   - Ratings alignment (books with similar ratings to those they've rated highly)

3. **Diversity Check**: Verify that recommendations aren't all from the same series or author.

### 6. Stopping the Container

When you're done testing:

```bash
docker-compose down
```

## For Testers

This section provides instructions for testers who have received this book recommender system as a packaged ZIP file.

### Package Contents

Your ZIP archive should contain:
```
MLOps_book_recommender_system/
├── src/              # Source code for the recommender system
├── models/           # Pre-trained model files
│   ├── user_item_matrix.npz
│   ├── book_features.npz
│   └── book_metadata.csv
├── data/             # Data directory (may contain sample data)
├── docker-compose.yml
└── README.md         # This file
```

### Prerequisites

1. **Docker and Docker Compose**
   - [Install Docker](https://docs.docker.com/get-docker/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

2. **Sufficient disk space**
   - At least 1GB for the Docker image and containers
   - Additional space for the model files

### Quick Start Testing

1. **Extract the ZIP archive** to a directory of your choice

2. **Verify model files** are present in the `models/` directory:
   - `user_item_matrix.npz`
   - `book_features.npz`
   - `book_metadata.csv`

3. **Start the Docker container**:
   ```bash
   cd MLOps_book_recommender_system
   docker-compose up -d
   ```

4. **Check if the container is running**:
   ```bash
   docker-compose ps
   ```
   You should see the `book-recommender-api` container with status "Up".

5. **Run the automated test script**:
   ```bash
   python src/api/test_api.py --server-url http://localhost:8000
   ```
   This script will test all API endpoints and report results.

6. **Explore the API documentation** at [http://localhost:8000/docs](http://localhost:8000/docs)

### What to Test

1. **User Recommendations**
   - Try recommendations for different users (e.g., user IDs 1, 10, 50)
   - URL: `http://localhost:8000/recommendations/user/1?limit=5`
   - Check if recommendations seem reasonable and diverse

2. **Similar Books**
   - Check if similar books make sense (e.g., books from the same series or by the same author)
   - URL: `http://localhost:8000/recommendations/similar-to/1?limit=5`
   - Book ID 1 is "Harry Potter and the Half-Blood Prince" - similar books should include other Harry Potter titles

3. **Book Details**
   - Verify detailed information for specific books
   - URL: `http://localhost:8000/books/1`

4. **Error Handling**
   - Test with invalid book IDs and user IDs
   - Verify appropriate error messages are returned

5. **Rating Submission** (requires authentication)
   - Test the rating submission endpoint with the API key specified in docker-compose.yml
   - ```bash
     curl -X POST http://localhost:8000/ratings \
       -H "Content-Type: application/json" \
       -H "X-API-Key: your-secret-api-key" \
       -d '{"user_id": 1, "book_id": 1, "rating": 5.0}'
     ```

### Stopping the Container

When you're done testing:
```bash
docker-compose down
```

### Reporting Issues

When reporting issues, please include:
1. The specific API endpoint that failed
2. Any error messages or unexpected output
3. Steps to reproduce the issue
4. Your operating system and Docker version

## Using DVC for Model and Data Files

This project uses Data Version Control (DVC) to manage large model and data files separately from Git. This allows us to version these files without bloating the Git repository.

### For Developers

#### Prerequisites
1. **Activate the virtual environment first!**
   ```bash
   # On Windows
   .\venv\Scripts\activate

   # On Unix/Mac
   source venv/bin/activate
   ```

2. **Install DVC and dependencies**
   ```bash
   # Make sure you're in an activated virtual environment
   pip install -r requirements.txt
   ```

#### Working with DVC

1. **Pull the model and data files** (after cloning the repository)
   ```bash
   # Make sure your virtual environment is activated
   dvc pull
   ```

2. **Add new or updated model files**
   ```bash
   # Make sure your virtual environment is activated
   
   # Add files to DVC
   dvc add models/new_model_file.npz
   
   # Commit the .dvc file to Git
   git add models.dvc
   git commit -m "Add new model file"
   
   # Push to DVC storage
   dvc push
   ```

3. **Update existing tracked files**
   ```bash
   # After changing files that are already tracked by DVC
   # Make sure your virtual environment is activated
   
   # Commit the changes
   dvc commit
   
   # Push to DVC storage
   dvc push
   ```

### For Testers

If you're testing this application, you'll need to get the model files from DVC:

1. **Clone the repository and create a virtual environment**
   ```bash
   git clone https://github.com/yourusername/MLOps_book_recommender_system.git
   cd MLOps_book_recommender_system
   
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Unix/Mac
   ```

2. **Install dependencies**
   ```bash
   # Make sure you're in an activated virtual environment
   pip install -r requirements.txt
   ```

3. **Pull the model and data files**
   ```bash
   # Make sure your virtual environment is activated
   dvc pull
   ```

   > Note: The DVC storage is currently configured locally. If you don't have access to the local storage, you may need to get the model files directly from the project maintainer.

4. **Run the API using Docker** (follow instructions in the "Running with Docker" section)

### Troubleshooting DVC

- **Files not downloading**: Make sure your virtual environment is activated
- **Access issues**: The DVC storage is local by default, so you need to have access to the local storage directory
- **Missing models**: Run `dvc status` to check which files are missing and then `dvc pull` to download them

## Data Requirements

To test the book recommender system, you'll need the following model files in the correct locations:

### Required Model Files

Place these files in the `models/` directory at the project root:

1. `user_item_matrix.npz`: Sparse matrix of user-item interactions
2. `book_features.npz`: Sparse matrix of book features 
3. `book_metadata.csv`: CSV file containing book metadata

### How to Get the Data

There are two ways to get the required data:

1. **Use pre-trained model files**: If you received this project with pre-trained model files, they should be in the `models/` directory.

2. **Train the model yourself**: If you don't have the model files, you'll need to train the model using:
   ```bash
   # First, download the raw data (if needed)
   python src/data/download_data.py
   
   # Process the data
   python src/data/make_dataset.py
   
   # Train the model
   python src/models/train_model.py
   ```

When running the containerized version, these model files are mounted from your local `models/` directory into the container.

## Running Locally

To run the API locally without Docker:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```
   cd src/api
   python run_api.py
   ```

3. Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

## Running with Docker

1. Ensure your model files are properly trained and available in the `models/` directory:
   - `user_item_matrix.npz`: Sparse matrix of user-item interactions
   - `book_features.npz`: Sparse matrix of book features
   - `book_metadata.csv`: CSV file containing book metadata including titles, authors, etc.

2. Build and start the Docker container:
   ```
   docker-compose up --build -d
   ```

3. Check if the container is running properly:
   ```
   docker-compose ps
   ```

4. Check the logs for any issues:
   ```
   docker-compose logs
   ```

5. The API will be available at [http://localhost:8000](http://localhost:8000)

6. Test the containerized API:
   ```
   python src/api/test_api.py --server-url http://localhost:8000
   ```

7. Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

> **Important Note About Model Files in Docker**: The Docker container does NOT include pre-trained model files. Instead, it mounts the `models/` directory from your local machine into the container. This means that you must have the required model files in your local `models/` directory before starting the container. See the "Data Requirements" section above for details on how to obtain these files.

## Troubleshooting

If you encounter issues during testing:

1. **Container not starting**: Check logs with `docker-compose logs`
2. **Model not loading**: Ensure model files exist in the correct location
3. **Authentication errors**: Verify the API key in the docker-compose.yml file
4. **Empty recommendations**: This might happen for users with very few ratings

## Development Workflow

For the best development experience, follow these steps:

1. **Test locally first**: Use `run_api.py` to start the API and `test_api.py` to test all endpoints.
2. **Fix any issues**: Debug and fix any problems in the local environment.
3. **Containerize**: Once everything works locally, build and test the Docker container.
4. **Deploy**: When the containerized version passes all tests, you can deploy it to your production environment.

## Security Notes

- In production, change the default API key by setting the `API_KEY` environment variable
- Consider adding SSL/TLS in production
- Add proper user authentication for a real-world deployment
