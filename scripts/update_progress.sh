#!/bin/bash
# update_progress.sh — kaguya_plan.md の進捗を自動更新
#
# 使用方法:
#   ./scripts/update_progress.sh CP-1.1 ✅ https://github.com/Carvlly/kaguya/pull/5
#   ./scripts/update_progress.sh CP-1.2 🟡 https://github.com/Carvlly/kaguya/pull/6

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <checkpoint-id> <status-emoji> <pr-url>"
    echo "Status emojis: ✅完了  🟡進行中  ❌失敗  ⬜未着手"
    exit 1
fi

CP_ID="$1"
STATUS="$2"
PR_URL="$3"
PLAN_FILE="kaguya_plan.md"

cd "$(git rev-parse --show-toplevel)"

if [ ! -f "$PLAN_FILE" ]; then
    echo "❌ kaguya_plan.md not found"
    exit 1
fi

# Replace the status and PR URL for the checkpoint
sed -i "s/| ${CP_ID} |.*| [^|]* | [^|]* |/| ${CP_ID} | ${STATUS} | ${PR_URL} |/" "$PLAN_FILE"

echo "✅ Updated ${CP_ID} → ${STATUS} (${PR_URL})"

# Show the updated line
grep "| ${CP_ID} |" "$PLAN_FILE"
