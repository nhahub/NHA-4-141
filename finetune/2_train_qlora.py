"""
Step 2: QLoRA fine-tuning of the base model on finetune/data/train.jsonl
(produced by 1_prepare_dataset.py).

Implements the three core QLoRA ingredients:
  - 4-bit NF4 quantization of the frozen base model (bnb_4bit_quant_type="nf4")
  - Double Quantization to shrink the quantization-constant overhead
    (bnb_4bit_use_double_quant=True)
  - Paged optimizer to avoid OOM during gradient checkpointing
    (optim="paged_adamw_8bit")
  - LoRA adapters on ALL linear layers, not just attention q/v — this is
    what the QLoRA paper found necessary to match full-finetuning quality.

Usage:
  pip install -r finetune/requirements-finetune.txt --break-system-packages
  python finetune/2_train_qlora.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTTrainer  # noqa: E402

from finetune import config  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    dataset_path = os.path.join(PROJECT_ROOT, config.DATASET_PATH)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run finetune/1_prepare_dataset.py first, "
            f"or point FT_DATASET_PATH at an existing train.jsonl."
        )

    print(f"Loading base model: {config.BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"  {len(dataset)} training examples")

    adapter_out = os.path.join(PROJECT_ROOT, config.ADAPTER_DIR)
    os.makedirs(adapter_out, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(adapter_out, "_checkpoints"),
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print(f"\nDone. LoRA adapter saved to {adapter_out}")
    print("Next: python finetune/3_merge_adapter.py")


if __name__ == "__main__":
    main()