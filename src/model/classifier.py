"""Safety classifier built on Qwen3.5.

Wraps a Qwen3.5 model for multimodal content safety classification
with Chain-of-Thought reasoning output.
"""

import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

from src.data.cot_templates import COT_SYSTEM_PROMPT, ANNOTATION_PROMPT, parse_cot_response


class SafetyClassifier:
    """Multimodal safety classifier using Qwen3.5 with CoT reasoning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        adapter_path: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )

        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    @classmethod
    def from_config(cls, config_path: str = "configs/model.yaml", **kwargs) -> "SafetyClassifier":
        """Load classifier from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_cfg = config["model"]
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}

        return cls(
            model_name=model_cfg["name"],
            dtype=dtype_map.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16),
            **kwargs,
        )

    @torch.inference_mode()
    def predict(self, image, text: str, max_new_tokens: int = 512) -> dict:
        """Run safety classification on an image + text pair.

        Args:
            image: PIL Image or path to image file.
            text: Text overlay or caption.
            max_new_tokens: Max tokens to generate.

        Returns:
            Dict with keys: classification, severity, reasoning (full CoT).
        """
        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": ANNOTATION_PROMPT + f'\nThe text reads: "{text}"'},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        response = self.processor.decode(new_tokens, skip_special_tokens=True)

        parsed = parse_cot_response(response)
        return {
            "classification": parsed["classification"],
            "severity": parsed["severity"],
            "reasoning": response,
            "parsed": parsed,
        }

    @torch.inference_mode()
    def predict_batch(self, samples: list[dict], max_new_tokens: int = 512) -> list[dict]:
        """Run classification on a batch of samples.

        Args:
            samples: List of dicts with 'image' and 'text' keys.
            max_new_tokens: Max tokens to generate per sample.

        Returns:
            List of prediction dicts.
        """
        return [self.predict(s["image"], s["text"], max_new_tokens) for s in samples]
