#!/usr/bin/env bash
# One-shot installer for LiveK12Bench dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m pip install --upgrade pip
python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "Done. Next steps:"
echo "  1. Configure LLM credentials (see README.md → 'LLM Configuration')"
echo "  2. Verify the install:"
echo "       python3 -c 'from litellm import completion; print(\"ok\")'"
