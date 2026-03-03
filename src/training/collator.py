"""Custom data collator for multimodal conversation fine-tuning.

Handles padding and batching of mixed image-text conversation samples
for Qwen3.5 training.
"""

from dataclasses import dataclass

import torch
from transformers import ProcessorMixin


@dataclass
class MultimodalCollator:
    """Collate multimodal conversation samples into training batches.

    Applies the chat template, processes images, and pads sequences
    for efficient batched training.
    """

    processor: ProcessorMixin
    max_length: int = 2048

    def __call__(self, examples: list[dict]) -> dict:
        texts = []
        images = []

        for ex in examples:
            messages = ex["messages"]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text)

            ex_images = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            ex_images.append(item["image"])
            images.append(ex_images if ex_images else None)

        batch = self.processor(
            text=texts,
            images=[img for img in images if img is not None],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100

        return batch
