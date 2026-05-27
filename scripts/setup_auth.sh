#!/bin/bash
# setup_auth.sh — GitHub認証のセットアップ
#
# 使用方法:
#   export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
#   ./scripts/setup_auth.sh
#
# または対話的に設定:
#   gh auth login --with-token <<< "ghp_xxxxxxxxxxxx"
#
# またはブラウザ認証:
#   gh auth login

set -e

REPO="Carvlly/kaguya"

echo "=== Kaguya GitHub Authentication Setup ==="

if [ -n "$GITHUB_TOKEN" ]; then
    echo "Found GITHUB_TOKEN environment variable"
    echo "$GITHUB_TOKEN" | gh auth login --with-token
    echo "✅ Authenticated with GITHUB_TOKEN"
elif gh auth status &>/dev/null; then
    echo "✅ Already authenticated as: $(gh api user --jq '.login')"
else
    echo "No authentication found. Please authenticate:"
    echo ""
    echo "Option 1: Set GITHUB_TOKEN environment variable"
    echo "  export GITHUB_TOKEN='ghp_your_token_here'"
    echo ""
    echo "Option 2: Interactive login"
    echo "  gh auth login"
    echo ""
    echo "Option 3: Device flow (no browser needed)"
    echo "  gh auth login --web"
    exit 1
fi

# Verify repo access
if gh repo view "$REPO" &>/dev/null; then
    echo "✅ Repository $REPO is accessible"
    # Check write access
    if gh api "repos/$REPO" --jq '.permissions.push' 2>/dev/null | grep -q "true"; then
        echo "✅ Write access confirmed — PR creation is possible"
    else
        echo "⚠️  Read-only access — fork & PR workflow will be used"
    fi
else
    echo "❌ Cannot access repository $REPO"
    exit 1
fi

# Configure git
git config user.name "Kaguya Developer"
git config user.email "kaguya-dev@users.noreply.github.com"
echo "✅ Git configured"
