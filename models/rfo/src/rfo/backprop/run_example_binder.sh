#!/usr/bin/env bash
# Run in the Foundry/RF3 environment from models/rfo. Supply a standalone
# optimizer YAML with input, output_path and checkpoint_path (see README.md).
set -euo pipefail
CONFIG="${1:?Usage: bash run_example_binder.sh /path/to/optimizer.yaml}"
CONFIG_DIR="$(cd "$(dirname "$CONFIG")" && pwd)"
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
python -m rfo.backprop.optimize --config_name "$CONFIG_NAME" --config_path "$CONFIG_DIR"
