#!/bin/bash
set -e  # Exit immediately on error

echo "🚀 Starting DVC container..."

# Create a .netrc file for GitHub authentication if GITHUB_TOKEN and GIT_USER_NAME are set
if [ -n "$GITHUB_TOKEN" ] && [ -n "$GIT_USER_NAME" ]; then
    cat <<EOF > /root/.netrc
machine github.com
  login $GIT_USER_NAME
  password $GITHUB_TOKEN
EOF
    chmod 600 /root/.netrc
    echo "🔧 .netrc file created for GitHub authentication."
fi

# Ensure log directory exists
mkdir -p /app/logs

# Remove any old completion flag
if [ -f "/app/logs/dvc_complete" ]; then
    echo "🧹 Removing stale DVC completion flag..."
    rm /app/logs/dvc_complete
fi

# Wait for the training stage to signal completion (using the dvc-tracked training_complete flag)
TRAINING_COMPLETE="/app/models/training_complete.dvc"
echo "⏳ Waiting for training process to fully complete..."
while [ ! -f "$TRAINING_COMPLETE" ]; do
    echo "🚨 Training not yet completed. Retrying in 20s..."
    sleep 20
done
echo "✅ Training stage completed. Proceeding with DVC operations."

# Verify that model artifacts exist
MODEL_PATH="/app/models/collaborative.pkl"
CONFIG_PATH="/app/models/collaborative_config.json"
echo "⏳ Verifying model artifacts..."
while [ ! -f "$MODEL_PATH" ] || [ ! -f "$CONFIG_PATH" ]; do
    echo "🚨 Model artifacts not yet available. Retrying in 20s..."
    sleep 20
done
echo "✅ Model artifacts detected."

# Initialize DVC if not already done
if [ ! -d "/app/.dvc" ]; then
    echo "⚙️ Initializing DVC..."
    dvc init --no-scm
fi

# Since the outputs of stage 'train_model' are managed by dvc.yaml,
# we use dvc commit on that stage to update the DVC cache without running the pipeline.
echo "🔄 Committing DVC outputs for stage 'train_model'..."
dvc commit train_model -f

# Configure Git user if environment variables are set
if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then
    echo "🔧 Configuring Git user..."
    git config --global user.name "$GIT_USER_NAME"
    git config --global user.email "$GIT_USER_EMAIL"
fi

# Initialize Git repository if needed
if [ ! -d "/app/.git" ]; then
    echo "⚠️ No Git repository found. Initializing..."
    git init
    git remote add origin "$GIT_REMOTE_URL"
fi

# Ensure we are on the correct branch
TARGET_BRANCH=${GIT_BRANCH:-main}  # Default to 'main' if GIT_BRANCH is not set
echo "🔄 Checking out the correct branch: $TARGET_BRANCH"
git fetch origin || echo "⚠️ Git fetch failed; might be a new repository"
git checkout -B "$TARGET_BRANCH" origin/"$TARGET_BRANCH" || git checkout -B "$TARGET_BRANCH"
git pull origin "$TARGET_BRANCH" || echo "⚠️ Git pull failed; might be a new branch"


git remote set-url origin "https://${GITHUB_TOKEN}@github.com/pepperumo/MLOps_book_recommender_system.git"

# (Optional) Add any non-pipeline files from /app/models that are not already in dvc.yaml.
# (Usually, you don’t want to add files already tracked by your DVC pipeline.)
# Only add files NOT already managed by the pipeline
for file in $(find /app/models -type f -not -path "*/\.*" | grep -v ".dvc"); do
    if ! grep -q "$file" dvc.yaml && ! grep -q "$(basename $file)" dvc.yaml; then
        echo "🔄 Adding non-pipeline file to DVC: $file"
        dvc add "$file" || echo "⚠️ Could not add $file, it might be managed by a pipeline"
    else
        echo "ℹ️ Skipping $file as it's managed by the pipeline"
    fi
done

if [[ "${SKIP_GIT_PUSH:-false}" != "true" ]]; then
  echo "📤 Committing updated DVC outputs to Git and pushing to branch: $TARGET_BRANCH..."
  git add .
  git commit -m "📅 Update collaborative model artifacts $(date +%Y-%m-%d)" || echo "⚠️ Nothing to commit"
  git push origin "$TARGET_BRANCH" || echo "⚠️ Push failed; check Git configuration"
else
  echo "⏩ Skipping Git push as requested"
fi

# Push updated artifacts to DVC remote storage
if dvc remote list | grep -q .; then
    echo "📤 Pushing updated artifacts to DVC remote storage..."
    dvc push || echo "⚠️ DVC push failed; check DVC remote configuration"
else
    echo "⚠️ No DVC remote configured, skipping push"
fi

# Signal completion
echo "✅ DVC tracking complete!"
touch /app/logs/dvc_complete

# Optionally keep the container alive for debugging
if [ "$1" = "keep-alive" ]; then
    echo "🐞 Keeping container alive for debugging."
    tail -f /dev/null
fi
