#!/bin/bash
set -euo pipefail  # Better error handling

echo "🚀 Starting DVC container..."

########################################
# GitHub – Personal Access Token Authentication
########################################
# Using GitHub's recommended approach for token auth in containers

if [[ -n "${GITHUB_TOKEN:-}" && -n "${GIT_USER_NAME:-}" ]]; then
  # Create a .netrc file which works better in container environments
  cat > /root/.netrc << EOF
machine github.com
login ${GIT_USER_NAME}
password ${GITHUB_TOKEN}
EOF
  chmod 600 /root/.netrc
  echo "🔧 GitHub token authentication configured with .netrc file."
fi

# Create directories if they don't exist
mkdir -p /app/logs /app/models /app/data

# Check if we should skip waiting for training (default is true in docker-compose.dvc.yml)
if [ "${SKIP_TRAINING_WAIT}" != "true" ]; then
    # This section only runs if SKIP_TRAINING_WAIT is explicitly set to false
    echo "⏳ Waiting for training process to complete..."
    
    # Wait for training complete flag
    while [ ! -f "/app/models/training_complete.dvc" ] && [ ! -f "/app/models/training_complete" ]; do
        echo "🚨 Training not yet completed. Waiting 20s before checking again..."
        sleep 20
    done
    echo "✅ Training completion flag detected."
    
    # Verify model artifacts
    while [ ! -f "/app/models/collaborative.pkl" ] || [ ! -f "/app/models/collaborative_config.json" ]; do
        echo "🚨 Model artifacts not available. Waiting 20s before checking again..."
        sleep 20
    done
    echo "✅ Model artifacts verified."
else
    echo "⏩ Skipping training wait as configured."
    
    # Just check if models exist without waiting
    if [ ! -f "/app/models/collaborative.pkl" ] || [ ! -f "/app/models/collaborative_config.json" ]; then
        echo "⚠️ Warning: Expected model artifacts not found. Will proceed anyway."
    else
        echo "✅ Model artifacts found."
    fi
fi

echo "🔍 Checking DVC configuration..."

# Initialize DVC if needed
if [ ! -d "/app/.dvc" ]; then
    echo "🏁 Initializing DVC..."
    cd /app && dvc init --no-scm
fi

# Configure DVC remote with DAGsHub credentials if available
if [ -n "$DAGSHUB_USER" ] && [ -n "$DAGSHUB_TOKEN" ]; then
    echo "🔄 Configuring DVC remote with DAGsHub credentials..."
    # Create credentials file for DVC
    mkdir -p /root/.dvc/tmp
    echo "https://$DAGSHUB_USER:$DAGSHUB_TOKEN@dagshub.com/$DAGSHUB_USER/MLOps_book_recommender_system.dvc" > /root/.dvc/tmp/auth
    chmod 600 /root/.dvc/tmp/auth
    echo "✅ DVC remote configured."
fi

# Configure Git
if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
    echo "🔧 Configuring Git..."
    git config --global user.name "$GIT_USER_NAME"
    git config --global user.email "$GIT_USER_EMAIL"
fi

# Add and push data
echo "📤 Adding files to DVC tracking..."

# Track model files
cd /app
if [ -f "models/collaborative.pkl" ]; then
    echo "📊 Tracking model: models/collaborative.pkl"
    dvc add models/collaborative.pkl || echo "⚠️ Could not add model file, might already be tracked"
fi

if [ -f "models/collaborative_config.json" ]; then
    echo "📊 Tracking config: models/collaborative_config.json"
    dvc add models/collaborative_config.json || echo "⚠️ Could not add config file, might already be tracked"
fi

# Track data directories
for dir in data/raw data/processed data/features; do
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        echo "📊 Tracking data directory: $dir"
        dvc add $dir || echo "⚠️ Could not add $dir, might already be tracked"
    fi
done

# Push to DVC remote
echo "📤 Pushing data to DVC remote storage..."
dvc push || echo "⚠️ DVC push failed, check remote configuration"

# Git operations if not skipped
if [[ "${SKIP_GIT_PUSH:-false}" != "true" ]]; then
    # Commit all changes including DVC files
    echo "📝 Committing all changes including DVC files to Git..."
    if [[ "${COMMIT_ALL_CHANGES:-false}" == "true" ]]; then
        # Commit all changes in the repository
        git add -A
        git commit -m "📊 Update project with DVC tracked files $(date +%Y-%m-%d)" || echo "⚠️ Nothing to commit"
    else
        # Only commit DVC files (default behavior)
        git add '*.dvc' .dvc/config || echo "⚠️ No DVC files to add"
        git commit -m "📊 Update DVC tracked files $(date +%Y-%m-%d)" || echo "⚠️ Nothing to commit"
    fi    
    
    # Push to Git if a token is available
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        echo "📤 Pushing to Git repository using token authentication..."
        
        # The token is already configured in the Git URL rewriting
        # Just need to make sure the remote has the correct URL
        git remote set-url origin "$GIT_HTTPS_URL"
        
        # Push to the remote repository (authentication handled by earlier config)
        git push origin "$GIT_BRANCH" || echo "⚠️ Git push failed, check credentials"
    else
        echo "⚠️ GITHUB_TOKEN not set, skipping Git push"
    fi
else
    echo "⏩ Git push skipped as requested"
fi

echo "✅ DVC operations completed!"

# Keep container alive if requested
if [[ ${1:-} = "keep-alive" ]]; then
    echo "🔄 Keeping container alive for debugging..."
    tail -f /dev/null
fi
