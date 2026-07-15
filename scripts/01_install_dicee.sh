#!/usr/bin/env bash
# Step 0 - environment setup (plain venv, no conda needed).
# Run this once, from the repo root, on a machine WITH internet access.
set -e

# 0) Make sure python3-venv / pip are present (Debian/Ubuntu/WSL)
if ! python3 -c "import venv" 2>/dev/null; then
    sudo apt update && sudo apt install -y python3-venv python3-pip
fi

# 1) Create and activate an isolated environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dice-embeddings (CPU build -- drop the --extra-index-url line
#    if you have a CUDA GPU and want GPU training instead)
pip install --upgrade pip
pip install dicee --extra-index-url https://download.pytorch.org/whl/cpu

# 3) Install the extra libs our attack/eval scripts need
pip install scikit-learn pandas numpy matplotlib

echo "Done. Verify with:  python -c 'import dicee; print(dicee.__version__)'"
echo "Remember: every new terminal session, re-run 'source .venv/bin/activate' before using this repo."
