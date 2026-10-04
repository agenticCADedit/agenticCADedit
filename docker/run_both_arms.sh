#!/usr/bin/env bash
#
# Runs the baseline (script) and our (mcp) arm sequentially with the same model.
#
# Usage: docker/run_both_arms.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for arm in script mcp; do
    echo "########## ${arm} ##########"
    "${HERE}/run_benchmark.sh" "${arm}"
done