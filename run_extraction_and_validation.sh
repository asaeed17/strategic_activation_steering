#!/bin/bash
# run_all_extraction.sh — Run extraction then validation for all variants
# Usage: bash run_all_extraction.sh 2>&1 | tee all_log.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "${SCRIPT_DIR}/run_extraction.sh"
bash "${SCRIPT_DIR}/run_validation.sh"
