#!/usr/bin/env bash
#
# Launches Audio Sample Generator.

set -e

deactivate > /dev/null 2>&1 || true

if [ ! -f ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

pip install -r requirements.txt

export PYTORCH_ENABLE_MPS_FALLBACK=1

streamlit run Getting_Started.py

