#!/bin/bash
# create_pr.sh — チェックポイント完了時のPR作成スクリプト
#
# 使用方法:
#   ./scripts/create_pr.sh CP-1.1 "AVX-512 SIMD dispatch mechanism"
#
# このスクリプトは以下を実行:
#   1. feature/CP-X.Y ブランチを作成
#   2. 変更をコミット
#   3. upstream (Carvlly/kaguya) にpush
#   4. PRを作成し、kaguya_plan.md の進捗を更新
#
# 前提: setup_auth.sh による認証が完了していること

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint-id> <description>"
    echo "Example: $0 CP-1.1 'AVX-512 SIMD dispatch mechanism'"
    exit 1
fi

CP_ID="$1"
DESCRIPTION="$2"
REPO="Carvlly/kaguya"
BRANCH="feature/${CP_ID,,}"
TITLE="[${CP_ID}] ${DESCRIPTION}"

# Validate gh auth
if ! gh auth status &>/dev/null; then
    echo "❌ Not authenticated. Run scripts/setup_auth.sh first."
    exit 1
fi

echo "=== Creating PR for ${CP_ID}: ${DESCRIPTION} ==="

cd "$(git rev-parse --show-toplevel)"

# Create and checkout branch
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
echo "✅ Branch: ${BRANCH}"

# Stage all changes
git add -A

# Check for changes
if git diff --cached --quiet; then
    echo "⚠️  No changes to commit. Make sure you have implemented the checkpoint."
    exit 1
fi

# Commit
COMMIT_MSG="${CP_ID}: ${DESCRIPTION}

Implements checkpoint ${CP_ID} for the Kaguya inference engine.

${DESCRIPTION}

See kaguya_plan.md for the full development roadmap."
git commit -m "$COMMIT_MSG"
echo "✅ Committed changes"

# Push (try direct, fallback to fork)
if git push -u origin "$BRANCH" 2>/dev/null; then
    echo "✅ Pushed to origin/${BRANCH}"
    HEAD_REPO="${REPO}"
else
    echo "ℹ️  Direct push failed, trying fork workflow..."
    gh repo fork "$REPO" --clone=false 2>/dev/null || true
    FORK_REPO=$(gh api user --jq '.login')/kaguya
    git remote add fork "https://github.com/${FORK_REPO}.git" 2>/dev/null || true
    git push -u fork "$BRANCH"
    echo "✅ Pushed to fork"
    HEAD_REPO="$FORK_REPO"
fi

# Create PR
PR_BODY="## 📋 ${CP_ID}: ${DESCRIPTION}

### 実装内容
${DESCRIPTION}

### テスト
- [ ] ユニットテストが通過
- [ ] ベンチマークでllama.cppと同等以上のパフォーマンス
- [ ] メモリリークなし (valgrind確認)

### 関連チェックポイント
kaguya_plan.md を参照

### 変更ファイル
$(git diff --stat HEAD~1)"

gh pr create \
    --repo "$REPO" \
    --title "$TITLE" \
    --body "$PR_BODY" \
    --head "${HEAD_REPO}:${BRANCH}" \
    --base main

echo ""
echo "=== ✅ PR created successfully ==="
echo "View: https://github.com/${REPO}/pulls"
