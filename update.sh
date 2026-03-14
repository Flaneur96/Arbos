#!/bin/bash
cd /Arbos
git fetch origin main --quiet 2>/dev/null
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [ "$LOCAL" != "$REMOTE" ]; then
    echo "$(date): Pulling updates..."
    git reset --hard origin/main
    /root/.local/bin/uv pip install -e . --quiet 2>/dev/null
    npm install -g @anthropic-ai/claude-code --quiet 2>/dev/null
    pm2 restart arbos
    echo "$(date): Restarted with $(git rev-parse --short HEAD)"
fi
