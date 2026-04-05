#!/bin/bash
# run_train.sh – Wrapper that exports Hugging Face credentials before training.
#
# Usage (same args as train.py):
#   bash run_train.sh --dataset mini --num_ways 5 --num_shots 1 \
#       --transductive True --pool_mode support --unet_mode addold
#
# The script reads /tmp/fsake_hf_config.json (written by the notebook Step 5A
# cell) and exports HF_USERNAME, HF_REPO_NAME, and HF_TOKEN so that the
# auto-push logic added to train.py activates automatically.

set -e

CFG="/tmp/fsake_hf_config.json"

if [ -f "$CFG" ]; then
    export HF_USERNAME=$(python3 -c "import json; d=json.load(open('$CFG')); print(d.get('HF_USERNAME',''))")
    export HF_REPO_NAME=$(python3 -c "import json; d=json.load(open('$CFG')); print(d.get('HF_REPO_ID','fsake-checkpoints').split('/')[-1])")
    # huggingface_hub stores the token at ~/.huggingface/token after login()
    TOKEN_FILE="$HOME/.huggingface/token"
    if [ -f "$TOKEN_FILE" ]; then
        export HF_TOKEN=$(cat "$TOKEN_FILE")
    fi
    echo "[HF] Credentials loaded: ${HF_USERNAME}/${HF_REPO_NAME}"
else
    echo "[HF] Config not found at $CFG — running without auto-push."
    echo "     Run the Step 5A cell in the notebook first."
fi

exec python3 train.py "$@"
