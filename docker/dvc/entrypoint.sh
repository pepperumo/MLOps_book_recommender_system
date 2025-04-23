#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting DVC container..."

########################################
# SSH key generation and GitHub setup
########################################
# Generate SSH keys inside the container if they don't exist

# Check volume permissions and content
echo "📁 Checking SSH directory..."
ls -la /root
ls -la /root/.ssh || echo "SSH directory doesn't exist yet"

mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Debug the contents of the SSH directory
echo "Files in /root/.ssh after mkdir:"
ls -la /root/.ssh

# Cache a copy of the key to avoid regeneration
if [[ -f "/app/.ssh-key-cache/id_ed25519" && ! -f "/root/.ssh/id_ed25519" ]]; then
  echo "🔄 Restoring SSH key from cache..."
  mkdir -p /app/.ssh-key-cache
  cp /app/.ssh-key-cache/id_ed25519* /root/.ssh/
  chmod 600 /root/.ssh/id_ed25519
fi

if [[ ! -f "/root/.ssh/id_ed25519" ]]; then
  echo "🔑 Generating SSH keys inside the container..."
  # Generate SSH key without passphrase
  ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N "" -C "pepperumo@gmail.com"
  chmod 600 /root/.ssh/id_ed25519
  
  # Cache the key in the project directory for persistence
  mkdir -p /app/.ssh-key-cache
  cp /root/.ssh/id_ed25519* /app/.ssh-key-cache/
  chmod 600 /app/.ssh-key-cache/id_ed25519
  
  # Display the public key for manual addition to GitHub
  echo "⚠️ IMPORTANT: Add this SSH public key to your GitHub account:"
  echo "===================="
  cat /root/.ssh/id_ed25519.pub
  echo "===================="
  echo "Then run the container again."
else
  echo "✅ Existing SSH key found at /root/.ssh/id_ed25519"
  echo "Public key fingerprint:"
  ssh-keygen -lf /root/.ssh/id_ed25519.pub
fi

# Add GitHub to known hosts to avoid prompt
ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null

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

for p in "${paths[@]}"; do
  if [[ -e "$p" ]]; then
    dvc commit "$p" 2>/dev/null || dvc add "$p" || echo "⚠️  $p already up‑to‑date"
  fi
done

echo "📤 Pushing data to DVC…"
dvc push || echo "⚠️  DVC push failed – check remote config"

########################################
# Git commit & push (optional)
########################################
if [[ "${SKIP_GIT_PUSH:-false}" != "true" ]]; then
  git add -A
  git commit -m "📊 Sync DVC artefacts $(date +%F)" || echo "ℹ️  Nothing to commit"
  
  # Use SSH for GitHub connection if available
  if [[ -f "/root/.ssh/id_rsa" || -f "/root/.ssh/id_ed25519" ]]; then
    echo "🔑 Using SSH for GitHub authentication"    # Make sure SSH key permissions are correct
    chmod 600 /root/.ssh/id_*
    # Add GitHub to known hosts to avoid prompt
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null
    # Set Git remote to SSH URL
    git remote set-url origin "git@github.com:pepperumo/MLOps_book_recommender_system.git"
    # Push using SSH
    git push origin "${GIT_BRANCH:-master}" || echo "⚠️  Git push failed - check SSH setup"
  else
    echo "⚠️ No SSH keys found - skipping Git push"
    echo "Please set up SSH authentication for GitHub"
  fi
else
  echo "⏩ Git push skipped."
fi

echo "✅ DVC operations completed!"

# Optional long‑running mode
if [[ "${1:-}" == "keep-alive" ]]; then
  tail -f /dev/null
fi