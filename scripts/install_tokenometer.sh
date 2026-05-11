#!/usr/bin/env bash
# install_tokenometer.sh — idempotent installer for the tokenometer CLI.
#
# What this does
# --------------
# Makes the tokenometer CLI available to `scripts/tokenometer_bridge.py` in
# one of three ways, in priority order:
#
#   1. If `tokenometer` is already on PATH (e.g. `npm install -g`), do
#      nothing. Idempotent — safe to re-run.
#   2. Otherwise, if a built sibling repo exists at
#      `../tokenometer/packages/cli/dist/index.js`, write a pointer file
#      `.tokenometer-cli-path` so the bridge knows where to find it.
#   3. Otherwise, if the sibling repo exists but is not built, attempt to
#      build it (`npm install && npm run build`), then write the pointer
#      file.
#
# If none of the above work, the script prints an install hint and exits 1
# so `make install` fails loudly rather than silently leaving the bridge
# broken.
#
# Why a sibling-repo pointer instead of `npm install`?
# ----------------------------------------------------
# The atlas repo is a Python project; it doesn't have a `package.json`.
# Adding one just to host a single npm dependency would muddy the project's
# tooling and force every contributor onto Node. The sibling-repo pattern
# matches how this project family is laid out
# (`personal/tokenometer/`, `personal/llm-tokens-atlas/`, ...).
#
# CI / fresh-machine paths
# ------------------------
# - **CI without npm secrets:** the workflow installs tokenometer globally
#   via `npm install -g tokenometer` before running `make install`.
# - **Local with sibling repo:** this script finds and pins the local build.
# - **Local without sibling repo:** this script suggests `npm install -g
#   tokenometer` and exits non-zero.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINFILE="$REPO_ROOT/.tokenometer-cli-path"
SIBLING_REPO="$(cd "$REPO_ROOT/.." 2>/dev/null && pwd)/tokenometer"
SIBLING_DIST="$SIBLING_REPO/packages/cli/dist/index.js"

log() { printf '[install_tokenometer] %s\n' "$*"; }

# (1) Already on PATH? Done.
if command -v tokenometer >/dev/null 2>&1; then
    log "tokenometer already on PATH ($(command -v tokenometer)). Nothing to do."
    # Clear any stale pinfile so the bridge prefers PATH.
    if [[ -f "$PINFILE" ]]; then
        log "Removing stale pinfile $PINFILE."
        rm -f "$PINFILE"
    fi
    exit 0
fi

# (2) Sibling build already present? Pin it.
if [[ -f "$SIBLING_DIST" ]]; then
    log "Found built sibling tokenometer at $SIBLING_DIST."
    printf 'node %s\n' "$SIBLING_DIST" > "$PINFILE"
    log "Wrote pinfile -> $PINFILE."
    exit 0
fi

# (3) Sibling source exists but not built? Build it.
if [[ -d "$SIBLING_REPO" ]] && [[ -f "$SIBLING_REPO/package.json" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
        log "Sibling repo found at $SIBLING_REPO but npm is not installed." >&2
        log "Install Node.js (>=20) from https://nodejs.org/ then re-run." >&2
        exit 1
    fi
    log "Sibling tokenometer at $SIBLING_REPO is not built. Building..."
    (
        cd "$SIBLING_REPO"
        npm install --no-audit --no-fund
        npm run build
    )
    if [[ -f "$SIBLING_DIST" ]]; then
        printf 'node %s\n' "$SIBLING_DIST" > "$PINFILE"
        log "Build complete; wrote pinfile -> $PINFILE."
        exit 0
    fi
    log "Build finished but $SIBLING_DIST is still missing." >&2
    exit 1
fi

# (4) Fallback: nothing usable. Suggest the npm global install.
cat >&2 <<'EOF'
[install_tokenometer] tokenometer CLI was not found.

Pick one of:

  # Recommended — installs from npm (registry: https://registry.npmjs.org/):
  npm install -g tokenometer

  # Local dev — clone the sibling repo next to this one:
  git clone https://github.com/faraa2m/tokenometer.git ../tokenometer
  (cd ../tokenometer && npm install && npm run build)
  bash scripts/install_tokenometer.sh   # re-run after the build lands

Then re-run `make install` (or this script directly).
EOF
exit 1
