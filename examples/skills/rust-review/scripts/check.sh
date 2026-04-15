#!/usr/bin/env bash
# rust-review/scripts/check.sh
# Run the MangoCode Rust QA pipeline. Copied into session workspace when skill loads.
# Usage: bash skills/rust-review/scripts/check.sh [crate]

set -euo pipefail

CRATE="${1:-}"
MANIFEST_FLAG=""
if [[ -n "$CRATE" ]]; then
    MANIFEST_FLAG="-p $CRATE"
fi

echo "=== cargo check ==="
cargo check $MANIFEST_FLAG 2>&1

echo ""
echo "=== cargo test ==="
cargo test $MANIFEST_FLAG 2>&1

echo ""
echo "=== cargo clippy ==="
cargo clippy $MANIFEST_FLAG -- -D warnings 2>&1

echo ""
echo "=== All checks passed ==="
