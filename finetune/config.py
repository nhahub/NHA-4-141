"""
Central config for the finetune/ pipeline. Change values here (or override
via environment variables) rather than editing the individual scripts.
"""
import os

# --- Base model to fine-tune ---
BASE_MODEL = os.getenv("FT_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# --- Paths (all relative to the finetune/ folder) ---
SOURCE_DOCS_DIR = os.getenv("FT_SOURCE_DOCS_DIR", "finetune/source_docs")
DATASET_PATH = os.getenv("FT_DATASET_PATH", "finetune/data/train.jsonl")
ADAPTER_DIR = os.getenv("FT_ADAPTER_DIR", "finetune/output/adapter")
MERGED_DIR = os.getenv("FT_MERGED_DIR", "finetune/output/merged")
GGUF_DIR = os.getenv("FT_GGUF_DIR", "finetune/output/gguf")

# --- Ollama connection (reused for dataset generation step) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Model Ollama uses to *generate* the synthetic QA training pairs from your
# documents. This can be a different (usually bigger/already-available) model
# than the one you are fine-tuning — e.g. keep "mistral" here even though
# BASE_MODEL above is Qwen2.5-1.5B.
GENERATOR_MODEL = os.getenv("FT_GENERATOR_MODEL", "qwen2.5:1.5b")

# --- Training hyperparameters (QLoRA defaults) ---
LORA_R = int(os.getenv("FT_LORA_R", 16))
LORA_ALPHA = int(os.getenv("FT_LORA_ALPHA", 32))
LORA_DROPOUT = float(os.getenv("FT_LORA_DROPOUT", 0.05))
LEARNING_RATE = float(os.getenv("FT_LEARNING_RATE", 2e-4))
NUM_EPOCHS = int(os.getenv("FT_NUM_EPOCHS", 3))
BATCH_SIZE = int(os.getenv("FT_BATCH_SIZE", 4))
GRAD_ACCUM_STEPS = int(os.getenv("FT_GRAD_ACCUM_STEPS", 4))
MAX_SEQ_LENGTH = int(os.getenv("FT_MAX_SEQ_LENGTH", 1024))

# Target modules for LoRA — QLoRA paper's key finding is that applying LoRA
# to ALL linear layers (not just q_proj/v_proj) is what closes the gap with
# full finetuning.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- Ollama model name for the finetuned model once converted to GGUF ---
OLLAMA_CUSTOM_MODEL_NAME = os.getenv("FT_OLLAMA_MODEL_NAME", "qwen-custom")

# --- GGUF quantization level ---
GGUF_QUANT_TYPE = os.getenv("FT_GGUF_QUANT_TYPE", "Q4_K_M")