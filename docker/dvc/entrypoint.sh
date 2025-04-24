#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting DVC container..."

########################################
# GitHub HTTPS Authentication Setup
########################################
echo "🔐 Setting up GitHub HTTPS authentication..."

# Configure Git to use the provided token for HTTPS authentication
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  # Hardcode the correct GitHub URL directly in the script
  GITHUB_URL="https://github.com/pepperumo/MLOps_book_recommender_system.git"
  GITHUB_DOMAIN="github.com"
  
  # Configure Git credential helper to store credentials in memory
  git config --global credential.helper store
  
  # Create a .netrc file with the GitHub token
  echo "machine ${GITHUB_DOMAIN}" > /root/.netrc
  echo "login ${GIT_USER_NAME:-pepperumo}" >> /root/.netrc
  echo "password ${GITHUB_TOKEN}" >> /root/.netrc
  chmod 600 /root/.netrc
  
  echo "✅ GitHub HTTPS authentication configured with token"
  

else
  echo "⚠️ GitHub token not provided. HTTPS authentication may not work."
fi

########################################
# Workspace housekeeping
########################################
########################################
########################################
mkdir -p /app/{logs,models,data}

########################################
# Optional wait for training artifacts
########################################
if [[ "${SKIP_TRAINING_WAIT}" != "true" ]]; then
  echo "⏳ Waiting for training artifacts..."
  while [[ ! -f /app/models/collaborative.pkl ]]; do
    sleep 20; echo "…still waiting";
  done
  echo "✅ Artifacts present.";
else
  echo "⏩ Skipping training wait as configured.";
fi

########################################
# DVC initialisation & remote setup
########################################
if [[ ! -d /app/.dvc ]]; then
  (cd /app && dvc init --no-scm)
fi

if [[ -n "${DAGSHUB_USER:-}" && -n "${DAGSHUB_TOKEN:-}" ]]; then
  echo "🔄 Configuring DVC remote for DagsHub…"
  dvc remote add   --force origin "https://dagshub.com/${DAGSHUB_USER}/MLOps_book_recommender_system.dvc"
  dvc remote modify       origin auth     basic
  dvc remote modify --local origin user     "${DAGSHUB_USER}"
  dvc remote modify --local origin password "${DAGSHUB_TOKEN}"
fi

########################################
# Git identity (needed for commits)
########################################
if [[ -n "${GIT_USER_NAME:-}" && -n "${GIT_USER_EMAIL:-}" ]]; then
  git config --global user.name  "${GIT_USER_NAME}"
  git config --global user.email "${GIT_USER_EMAIL}"
fi

########################################
# Update tracked artefacts without re‑adding pipeline outputs
########################################
cd /app

declare -a paths=(
  models/collaborative.pkl
  models/collaborative_config.json
  data/raw
  data/processed
  data/features
)

# Force commit all tracked files to ensure latest changes are captured
echo "🔄 Forcing DVC to commit latest versions of tracked files..."
for p in "${paths[@]}"; do
  if [[ -e "$p" ]]; then
    dvc commit -f "$p" || dvc add "$p" || echo "⚠️ Could not update $p"
  fi
done

echo "📤 Pushing data to DVC…"
dvc push -v || echo "⚠️  DVC push failed – check remote config"

########################################
# Git commit & push (optional)
########################################
if [[ "${SKIP_GIT_PUSH:-false}" != "true" ]]; then
  git add -A
  git commit -m "📊 Sync DVC artefacts $(date +%F)" || echo "ℹ️  Nothing to commit"
  
  # Use HTTPS with token authentication
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo "🔑 Using HTTPS with token for GitHub authentication"
    
    # Set up Git credentials using the GitHub token in the correct format
    GITHUB_URL="https://github.com/pepperumo/MLOps_book_recommender_system.git"
    
    # Properly configure Git to use the token
    git config --global credential.helper 'store --file=/tmp/git-credentials'
    echo "https://${GITHUB_TOKEN}:x-oauth-basic@github.com" > /tmp/git-credentials
    chmod 600 /tmp/git-credentials
    
    # Set the remote URL (without credentials embedded)
    git remote set-url origin "${GITHUB_URL}"
    
    echo "📌 Pushing to GitHub using HTTPS authentication with token"
    
    # Push using the stored credentials
    git push origin "${GIT_BRANCH:-master}" || echo "⚠️  Git push failed - check GitHub token"
    
    # Clean up credentials
    rm -f /tmp/git-credentials
  else
    echo "⚠️ No GitHub token available - skipping Git push"
    echo "Please set the GITHUB_TOKEN environment variable"
  fi
else
  echo "⏩ Git push skipped."
fi

echo "✅ DVC operations completed!"

# Optional long‑running mode
if [[ "${1:-}" == "keep-alive" ]]; then
  tail -f /dev/null
fi