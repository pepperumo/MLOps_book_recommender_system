#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting DVC container..."

########################################
########################################
# GitHub – Personal‑Access‑Token over HTTPS
########################################
# Git ignores “.netrc” unless a helper tells it to use it, so we
# configure the built‑in *store* helper and write the credentials once.
# The helper reads the token for every `git push` / `git fetch`.

if [[ -n "${GITHUB_TOKEN:-}" && -n "${GIT_USER_NAME:-}" ]]; then
  git config --global credential.helper 'store --file /root/.git-credentials'
  printf "https://%s:%s@github.com
" "$GIT_USER_NAME" "$GITHUB_TOKEN" > /root/.git-credentials
  chmod 600 /root/.git-credentials
  echo "🔧  GitHub token authentication configured (credential.store)."
fi

########################################
if [[ -n "${GITHUB_TOKEN:-}" && -n "${GIT_USER_NAME:-}" ]]; then
  cat <<EOF > /root/.netrc
machine github.com
login ${GIT_USER_NAME}
password ${GITHUB_TOKEN}
machine api.github.com
login ${GIT_USER_NAME}
password ${GITHUB_TOKEN}
EOF
  chmod 600 /root/.netrc
  echo "🔧  GitHub token authentication configured (.netrc)."
fi

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
  git remote set-url origin "${GIT_HTTPS_URL}"
  git push origin "${GIT_BRANCH:-master}" || echo "⚠️  Git push failed"
else
  echo "⏩ Git push skipped."
fi

echo "✅ DVC operations completed!"

# Optional long‑running mode
if [[ "${1:-}" == "keep-alive" ]]; then
  tail -f /dev/null
fi