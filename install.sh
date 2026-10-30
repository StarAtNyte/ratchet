#!/usr/bin/env sh
# Install Ratchet as a global npm command (same approach as Evidra's installer).
#   curl -fsSL -H "Authorization: Bearer $(gh auth token)" \
#     https://raw.githubusercontent.com/StarAtNyte/ratchet/main/install.sh | sh
# Env: RATCHET_REPO (default StarAtNyte/ratchet), RATCHET_REF (branch/tag, default main).
set -eu

REPO="${RATCHET_REPO:-StarAtNyte/ratchet}"
REF="${RATCHET_REF:-main}"
MIN_NODE_MAJOR=22

if ! command -v node >/dev/null 2>&1; then
  echo "Ratchet requires Node.js ${MIN_NODE_MAJOR}+. Install it with nvm first:"
  echo "  nvm install ${MIN_NODE_MAJOR} && nvm use ${MIN_NODE_MAJOR}"
  exit 1
fi
if ! node -e "process.exit(Number(process.versions.node.split('.')[0]) >= ${MIN_NODE_MAJOR} ? 0 : 1)"; then
  echo "Ratchet requires Node.js ${MIN_NODE_MAJOR}+ (found $(node --version))."
  exit 1
fi
command -v git >/dev/null 2>&1 || { echo "Ratchet requires git to install from source."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "Ratchet requires npm."; exit 1; }

INSTALL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ratchet-install.XXXXXX")
cleanup() { rm -rf "$INSTALL_DIR"; }
trap cleanup EXIT
trap 'printf "\nInstallation interrupted.\n" >&2; exit 130' INT
trap 'printf "\nInstallation terminated.\n" >&2; exit 143' TERM

run_stage() {
  STAGE_LABEL=$1
  shift
  printf '\n→ %s\n' "$STAGE_LABEL"
  STAGE_STATUS=0
  "$@" >"$INSTALL_DIR/stage.log" 2>&1 || STAGE_STATUS=$?
  if [ "$STAGE_STATUS" -ne 0 ]; then
    printf '✗ %s failed (exit %s)\n' "$STAGE_LABEL" "$STAGE_STATUS" >&2
    cat "$INSTALL_DIR/stage.log" >&2
    exit "$STAGE_STATUS"
  fi
  printf '✓ %s complete\n' "$STAGE_LABEL"
}

# Public repos clone over https; private ones need the gh CLI or an SSH key.
fetch() {
  git clone --quiet --depth 1 --branch "$REF" "https://github.com/$REPO.git" "$INSTALL_DIR/ratchet" 2>/dev/null \
    || { command -v gh >/dev/null 2>&1 && gh repo clone "$REPO" "$INSTALL_DIR/ratchet" -- --quiet --depth 1 --branch "$REF"; } \
    || git clone --quiet --depth 1 --branch "$REF" "git@github.com:$REPO.git" "$INSTALL_DIR/ratchet"
}

run_stage "1/4 Downloading Ratchet" fetch
cd "$INSTALL_DIR/ratchet"
run_stage "2/4 Packaging CLI" npm pack --ignore-scripts --quiet
PACKAGE_VERSION=$(node -p "require('./package.json').version")
TARBALL="$INSTALL_DIR/ratchet/ratchet-$PACKAGE_VERSION.tgz"

# Use the default global prefix when writable, otherwise fall back to ~/.local.
PREFIX=$(npm prefix --global)
if [ ! -w "$PREFIX" ] && [ ! -w "$PREFIX/lib" ] 2>/dev/null; then
  PREFIX="$HOME/.local"
  mkdir -p "$PREFIX"
fi
# A link from an older checkout-style install would conflict with npm's bin link.
OLD="$HOME/.local/bin/ratchet"
if [ -L "$OLD" ] && readlink "$OLD" | grep -q 'bin/ratchet.mjs$'; then rm -f "$OLD"; fi
run_stage "3/4 Installing global command" npm install --global --prefix "$PREFIX" "$TARBALL"
run_stage "4/4 Verifying installation" "$PREFIX/bin/ratchet" --version

echo "Ratchet installed globally."
case ":$PATH:" in
  *":$PREFIX/bin:"*) command -v ratchet >/dev/null 2>&1 || echo "Stale shell cache. Run: hash -r 2>/dev/null || rehash" ;;
  *) echo "Add $PREFIX/bin to your PATH to run ratchet:"; echo "  export PATH=\"$PREFIX/bin:\$PATH\"" ;;
esac
for c in claude codex agy; do
  if command -v "$c" >/dev/null 2>&1; then echo "found harness CLI: $c"; else echo "missing harness CLI: $c (optional)"; fi
done
printf '\nStart using Ratchet:\n'
printf '  ratchet --mock          Simulated agents, no quota (try the UI)\n'
printf '  ratchet <folder>        Open the web terminal on a workspace\n'
printf '  ratchet --help          Options\n'
