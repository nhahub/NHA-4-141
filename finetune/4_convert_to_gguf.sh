#!/usr/bin/env bash
# Step 4: Convert the merged HF model (finetune/output/merged) to GGUF and
# quantize it so it can run in Ollama.
#
# Usage:
#   bash finetune/4_convert_to_gguf.sh
#
# Requires llama.cpp. If you don't have it yet, this script clones it once
# into finetune/llama.cpp/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MERGED_DIR="$PROJECT_ROOT/finetune/output/merged"
GGUF_DIR="$PROJECT_ROOT/finetune/output/gguf"
LLAMA_CPP_DIR="$PROJECT_ROOT/finetune/llama.cpp"

# Quantization level — Q4_K_M is a good default balance of size/quality.
# Override with: QUANT_TYPE=Q5_K_M bash finetune/4_convert_to_gguf.sh
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"

if [ ! -d "$MERGED_DIR" ]; then
  echo "Error: $MERGED_DIR not found. Run finetune/3_merge_adapter.py first."
  exit 1
fi

mkdir -p "$GGUF_DIR"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "Cloning llama.cpp into $LLAMA_CPP_DIR ..."
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
  pip install -r "$LLAMA_CPP_DIR/requirements.txt" --break-system-packages
  echo "Building llama-quantize binary..."
  cmake -B "$LLAMA_CPP_DIR/build" -S "$LLAMA_CPP_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j
fi

echo "Converting merged HF model to f16 GGUF..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$GGUF_DIR/model-f16.gguf" \
  --outtype f16

echo "Quantizing to $QUANT_TYPE..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
  "$GGUF_DIR/model-f16.gguf" \
  "$GGUF_DIR/model-$QUANT_TYPE.gguf" \
  "$QUANT_TYPE"

echo
echo "Done. GGUF file ready at: $GGUF_DIR/model-$QUANT_TYPE.gguf"
echo "Next: bash finetune/5_create_ollama_model.sh"