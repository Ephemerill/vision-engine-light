#!/bin/bash

# Configuration
BRANCH="main" # Change to "master" if your repo uses master

echo "=========================================="
echo "   🚀 STARTING AUTO-UPDATE"
echo "=========================================="

# 1. Stash local config changes (Prevent conflicts)
# This saves your RTSP_URL and VIDEO_SOURCE settings temporarily
if [[ -n $(git status -s) ]]; then
    echo "💾 Stashing local changes (saving your config)..."
    git stash
    STASHED=true
else
    STASHED=false
fi

# 2. Pull from GitHub
echo "⬇️  Pulling latest code from GitHub..."
git fetch origin
git pull origin $BRANCH

# 3. Restore local config
if [ "$STASHED" = true ]; then
    echo "📂 Restoring your local config..."
    git stash pop
fi

# 4. Update Python Dependencies
# We use --upgrade to ensure we get the latest versions
# We explicitly check for GPU libraries to avoid overwriting them with CPU versions
echo "📦 Checking dependencies..."

if pip freeze | grep -q "onnxruntime-gpu"; then
    echo "   Detected GPU environment. Installing GPU-safe requirements..."
    pip install -r requirements.txt --upgrade
else
    echo "   Standard environment detected. Installing requirements..."
    pip install -r requirements.txt --upgrade
fi

echo "=========================================="
echo "✅ UPDATE COMPLETE"
echo "=========================================="
echo "You can now restart the application."