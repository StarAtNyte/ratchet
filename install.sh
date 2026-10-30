#!/usr/bin/env bash
# Install ratchet. Works from a checkout or piped from curl:
#   curl -fsSL -H "Authorization: Bearer $(gh auth token)" \
#     https://raw.githubusercontent.com/StarAtNyte/ratchet/main/install.sh | bash
# Env: RATCHET_DIR (install location, default ~/.local/share/ratchet), RATCHET_REPO (default StarAtNyte/ratchet).
set -euo pipefail

REPO="${RATCHET_REPO:-StarAtNyte/ratchet}"
DEST="${RATCHET_DIR:-$HOME/.local/share/ratchet}"

command -v node >/dev/null || { echo "node >= 22 required" >&2; exit 1; }
[ "$(node -p 'process.versions.node.split(".")[0]')" -ge 22 ] || { echo "node >= 22 required (have $(node -v))" >&2; exit 1; }
command -v git >/dev/null || { echo "git required" >&2; exit 1; }

SRC="${BASH_SOURCE[0]:-}"
if [ -n "$SRC" ] && [ -f "$(dirname "$SRC")/package.json" ] && [ -f "$(dirname "$SRC")/bin/ratchet.mjs" ]; then
  DIR="$(cd "$(dirname "$SRC")" && pwd)"            # run from a checkout
else
  DIR="$DEST"                                        # piped from curl: fetch the repo
  if [ -d "$DIR/.git" ]; then
    echo "updating $DIR"
    git -C "$DIR" pull --ff-only --quiet
  else
    echo "cloning $REPO into $DIR"
    mkdir -p "$(dirname "$DIR")"
    if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
      gh repo clone "$REPO" "$DIR" -- --quiet
    else
      git clone --quiet "git@github.com:$REPO.git" "$DIR" || git clone --quiet "https://github.com/$REPO.git" "$DIR"
    fi
  fi
fi

cd "$DIR"
npm install --no-audit --no-fund --silent

mkdir -p "$HOME/.local/bin"
ln -sf "$DIR/bin/ratchet.mjs" "$HOME/.local/bin/ratchet"
chmod +x "$DIR/bin/ratchet.mjs"

echo "installed: $HOME/.local/bin/ratchet -> $DIR/bin/ratchet.mjs"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "add to PATH:  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;; esac
for c in claude codex agy; do command -v "$c" >/dev/null && echo "found harness CLI: $c" || echo "missing harness CLI: $c (optional)"; done
echo "try:  ratchet --mock      # simulated agents, no quota"
echo "      ratchet ~/my/project"
