#!/bin/bash
set -e

version="$1"
if [ -z "$version" ]; then
    echo "Usage: $0 <version>" >&2
    exit 1
fi

# Save current symlink
original=$(readlink ~/.local/bin/claude)

# Install (changes symlink)
curl -fsSL https://claude.ai/install.sh | bash -s "$version" >/dev/null 2>&1

# Restore immediately
ln -sf "$original" ~/.local/bin/claude

# Return binary path
echo "$HOME/.local/share/claude/versions/$version"