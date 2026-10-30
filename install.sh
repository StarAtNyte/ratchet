#!/usr/bin/env bash
# Install ratchet from a checkout: npm deps + `ratchet` on PATH (~/.local/bin).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v node >/dev/null || { echo "node >= 22 required" >&2; exit 1; }
[ "$(node -p 'process.versions.node.split(".")[0]')" -ge 22 ] || { echo "node >= 22 required (have $(node -v))" >&2; exit 1; }

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
