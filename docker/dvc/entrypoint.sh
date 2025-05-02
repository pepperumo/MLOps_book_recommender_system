#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting DVC container..."

########################################
# Workspace housekeeping
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

echo "✅ DVC operations completed!"

# Optional long‑running mode
if [[ "${1:-}" == "keep-alive" ]]; then
  tail -f /dev/null
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

echo "✅ DVC operations completed!"

