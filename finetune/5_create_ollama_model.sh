#!/usr/bin/env bash
# Step 5: Register the quantized GGUF model with Ollama so the Streamlit app
# can call it just like it currently calls "mistral".
#
# Usage:
#   bash finetune/5_create_ollama_model.sh
#
# Then update your .env:
#   OLLAMA_MODEL=qwen-custom
# (or whatever FT_OLLAMA_MODEL_NAME you set)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
MODEL_NAME="${FT_OLLAMA_MODEL_NAME:-qwen-custom}"
GGUF_FILE="$PROJECT_ROOT/finetune/output/gguf/model-$QUANT_TYPE.gguf"
MODELFILE="$PROJECT_ROOT/finetune/output/gguf/Modelfile"

if [ ! -f "$GGUF_FILE" ]; then
  echo "Error: $GGUF_FILE not found. Run finetune/4_convert_to_gguf.sh first."
  exit 1
fi

cat > "$MODELFILE" <<EOF
FROM $GGUF_FILE

# Qwen2.5-Instruct chat template
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
EOF

echo "Creating Ollama model '$MODEL_NAME'..."
ollama create "$MODEL_NAME" -f "$MODELFILE"

echo
echo "Done. Test it with: ollama run $MODEL_NAME"
echo "Then in your .env file, set: OLLAMA_MODEL=$MODEL_NAME"