#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDSYSTEM_DIR="${BEP_BUILDSYSTEM_PATH:-$SCRIPT_DIR/third_party/BuildSystem}"
INTERFACE="$BUILDSYSTEM_DIR/scripts/interface.sh"

if [[ ! -f "$INTERFACE" ]]; then
  echo "[BEP][ERROR] BuildSystem interface not found: $INTERFACE" >&2
  exit 1
fi

exec "$INTERFACE" "$SCRIPT_DIR/buildsystem.xml" "$SCRIPT_DIR"
