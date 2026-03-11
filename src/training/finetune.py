"""LoRA fine-tuning for Qwen3.5 on content safety with CoT annotations.

Uses Unsloth for memory-efficient fine-tuning on a single A100.
Falls back to HuggingFace PEFT + TRL if Unsloth is not available.
"""

import json
import yaml
import argparse
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

from src.data.cot_templates import ANNOTATION_PROMPT
from src.data.preprocessing import create_splits


def load_configs(
    model_config: str = "configs/model.yaml",
    train_config: str = "configs/train.yaml",
) -> tuple[dict, dict]:
    with open(model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(train_config) as f:
        train_cfg = yaml.safe_load(f)
    return model_cfg, train_cfg


def setup_model(model_cfg: dict) -> tuple:
    """Load base model and processor, apply LoRA."""
    model_name = model_cfg["model"]["name"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(model_cfg["model"].get("dtype", "bfloat16"), torch.bfloat16)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
    )

    lora_cfg = model_cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def load_training_data(data_path: str = "data/train_cot.jsonl") -> Dataset:
    """Load CoT annotations and re-associate with images from the dataset.

    The JSONL stores text-only CoT records with an index field.
    Images are loaded from the original HuggingFace dataset.
    """
    # Load the original dataset for images
    hf_ds = load_dataset("Multimodal-Fatima/Hatefulmemes_train", split="train")
    splits = create_splits([{"label": row["label"]} for row in hf_ds])
    train_indices = list(range(len(splits["train"])))

    # Load CoT annotations
    with open(data_path) as f:
        cot_records = [json.loads(line) for line in f]

    # Build training samples with conversation format
    samples = []
    for record in cot_records:
        idx = record["index"]
        row = hf_ds[idx]
        prompt = ANNOTATION_PROMPT + f'\nThe text reads: "{record["text"]}"'
        # Store as text-only conversation (images handled by collator)
        samples.append({
            "messages": json.dumps([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": record["cot"]},
            ]),
        })

    return Dataset.from_list(samples)


def train(
    model_config: str = "configs/model.yaml",
    train_config: str = "configs/train.yaml",
    data_path: str = "data/train_cot.jsonl",
):
    """Run LoRA fine-tuning."""
    model_cfg, train_cfg = load_configs(model_config, train_config)
    model, processor = setup_model(model_cfg)
    dataset = load_training_data(data_path)

    t = train_cfg["training"]

    training_args = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_seq_length=t["max_seq_length"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        bf16=t["bf16"],
        report_to=t["report_to"],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(Path(t["output_dir"]) / "final")
    print(f"Model saved to {t['output_dir']}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--data-path", default="data/train_cot.jsonl")
    args = parser.parse_args()
    train(args.model_config, args.train_config, args.data_path)
