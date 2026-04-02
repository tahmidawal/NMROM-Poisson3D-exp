#!/bin/bash

# git-push.sh - Flexible git push script
# Usage:
#   ./git-push.sh                           # Push to current branch with "Updates" message
#   ./git-push.sh - "commit message"        # Push to current branch with custom message
#   ./git-push.sh branch-name               # Push to branch (create if new) with "Updates" message
#   ./git-push.sh branch-name "commit msg"  # Push to branch (create if new) with custom message

BRANCH="$1"
COMMIT_MSG="$2"

# Default commit message
DEFAULT_MSG="Updates"

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)

# Case 1: No arguments - push to current branch with default message
if [ -z "$BRANCH" ]; then
    BRANCH="$CURRENT_BRANCH"
    COMMIT_MSG="$DEFAULT_MSG"

# Case 2: Branch is "-" - stay on current branch, use provided message or default
elif [ "$BRANCH" = "-" ]; then
    BRANCH="$CURRENT_BRANCH"
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="$DEFAULT_MSG"
    fi

# Case 3: Branch name provided
else
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="$DEFAULT_MSG"
    fi
fi

echo "Branch: $BRANCH"
echo "Commit message: $COMMIT_MSG"
echo ""

# Check if branch exists locally
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    # Branch exists locally
    if [ "$BRANCH" != "$CURRENT_BRANCH" ]; then
        echo "Switching to existing branch: $BRANCH"
        git checkout "$BRANCH"
    fi
else
    # Branch doesn't exist - create it
    echo "Creating new branch: $BRANCH"
    git checkout -b "$BRANCH"
fi

# Stage all changes
git add .

# Commit
git commit -m "$COMMIT_MSG"

# Push (set upstream if new branch)
if git ls-remote --exit-code --heads origin "$BRANCH" > /dev/null 2>&1; then
    # Remote branch exists
    git push
else
    # Remote branch doesn't exist - set upstream
    git push -u origin "$BRANCH"
fi

echo ""
echo "Done! Pushed to $BRANCH"
