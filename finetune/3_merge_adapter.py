"""
Step 3: Merge the trained LoRA adapter back into the base model, producing a
single standalone HF model directory that can be converted to GGUF.

Usage:
  python finetune/3_merge_adapter.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

from finetune import config  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    adapter_dir = os.path.join(PROJECT_ROOT, config.ADAPTER_DIR)
    merged_dir = os.path.join(PROJECT_ROOT, config.MERGED_DIR)

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"Adapter not found at {adapter_dir}. Run finetune/2_train_qlora.py first."
        )

    print(f"Loading base model in full precision: {config.BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # merging doesn't need a GPU and avoids VRAM issues
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    print(f"Loading adapter from: {adapter_dir}")
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter weights into base model...")
    merged_model = merged_model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    print(f"\nDone. Merged model saved to {merged_dir}")
    print("Next: finetune/4_convert_to_gguf.sh")


if __name__ == "__main__":
    main()