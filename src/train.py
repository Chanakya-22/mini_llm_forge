"""Enterprise-grade training pipeline for ChatML-aware QLoRA fine-tuning.

This module defines an object-oriented training workflow for loading a
conversational dataset, configuring a tokenizer with a strict ChatML template,
setting up QLoRA with 4-bit quantization, and running supervised fine-tuning
with Hugging Face TRL's ``SFTTrainer``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.core.config import settings
from src.core.logger import logger


class LLMForgeTrainer:
    """Coordinate the full ChatML-based fine-tuning pipeline.

    The trainer handles dataset loading, tokenizer preparation, QLoRA
    configuration, model loading, training execution, and adapter export.
    """

    def __init__(self, dataset_path: str = "data/chatml_data.jsonl", output_dir: Optional[str] = None) -> None:
        """Initialize the trainer with dataset and output paths.

        Args:
            dataset_path: Path to the JSONL training dataset.
            output_dir: Optional output directory for the trained adapter. If not provided,
                uses settings.adapter_path or defaults to "adapters/default".

        Raises:
            ValueError: If no valid output directory can be determined.
        """
        self.dataset_path = Path(dataset_path)

        if output_dir:
            self.output_dir = Path(output_dir)
        elif settings.adapter_path:
            self.output_dir = Path(settings.adapter_path)
        else:
            self.output_dir = Path("adapters/default")
            logger.warning("No adapter_path configured; using default output directory: %s", self.output_dir)

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.trainer: Optional[SFTTrainer] = None
        self.dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    def _setup_logging(self) -> None:
        """Configure the module logger for training execution."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    def _load_dataset(self) -> Tuple[Dataset, Dataset]:
        """Load and split the conversational JSONL dataset.

        Returns:
            A tuple of (train_dataset, eval_dataset).

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset does not contain a ``messages`` column.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Training data not found at {self.dataset_path}")

        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        if "messages" not in dataset.column_names:
            raise ValueError("The dataset must contain a 'messages' column for ChatML training.")

        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        self.dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]
        logger.info("Loaded %s training examples and %s validation examples.", len(self.dataset), len(self.eval_dataset))
        return self.dataset, self.eval_dataset

    def _configure_tokenizer(self) -> Any:
        """Load and configure the tokenizer for ChatML formatting.

        Returns:
            A configured Hugging Face tokenizer instance.
        """
        tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)
        chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

        if getattr(tokenizer, "chat_template", None) != chat_template:
            tokenizer.chat_template = chat_template

        for special_token in ["<|im_start|>", "<|im_end|>"]:
            if special_token not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})

        tokenizer.pad_token = tokenizer.eos_token or "<|im_end|>"
        tokenizer.padding_side = "right"
        return tokenizer

    def _build_quantization_config(self) -> BitsAndBytesConfig:
        """Create a 4-bit QLoRA quantization configuration.

        Returns:
            A configured ``BitsAndBytesConfig`` instance.
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _build_lora_config(self) -> LoraConfig:
        """Create a targeted LoRA configuration for causal language modeling.

        Returns:
            A ``LoraConfig`` instance.
        """
        return LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

    def _build_training_config(self) -> SFTConfig:
        """Create the SFT training configuration for TRL.

        Returns:
            A configured ``SFTConfig`` instance.
        """
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            logging_steps=1,
            learning_rate=2e-4,
            bf16=bf16_supported,
            fp16=not bf16_supported and torch.cuda.is_available(),
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=50,
            save_steps=50,
            optim="paged_adamw_32bit",
            max_length=512,  # Correctly updated for TRL > 0.15
            packing=False,
            lr_scheduler_type="cosine",
        )

    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load the base model and tokenizer for QLoRA training.

        Returns:
            A tuple containing the model and tokenizer.
        """
        quantization_config = self._build_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            settings.base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
        tokenizer = self._configure_tokenizer()

        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        model.config.use_cache = False
        model.train()
        return model, tokenizer

    def _build_trainer(self, model: Any, tokenizer: Any) -> SFTTrainer:
        """Construct the ``SFTTrainer`` instance.

        Args:
            model: The loaded causal language model.
            tokenizer: The configured tokenizer.

        Returns:
            A configured ``SFTTrainer`` instance.
        """
        return SFTTrainer(
            model=model,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=tokenizer,
            args=self._build_training_config(),
            peft_config=self._build_lora_config(),
        )

    def train(self) -> None:
        """Execute the full training workflow.

        Raises:
            RuntimeError: If training fails unexpectedly.
        """
        self._setup_logging()
        logger.info("Starting ChatML training pipeline...")

        try:
            self.dataset = self._load_dataset()
            self.model, self.tokenizer = self._load_model_and_tokenizer()
            self.trainer = self._build_trainer(self.model, self.tokenizer)

            logger.info("Training started. This may take a while depending on hardware.")
            self.trainer.train()

            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))
            logger.info("Training complete. Adapter and tokenizer saved to %s.", self.output_dir)
        except torch.OutOfMemoryError as exc:
            logger.exception("Training failed due to out-of-memory.")
            raise RuntimeError("Training failed due to insufficient GPU memory.") from exc
        except Exception as exc:
            logger.exception("Training pipeline failed.")
            raise RuntimeError(f"Training pipeline failed: {exc}") from exc


def train() -> None:
    """Entry point for the training pipeline."""
    trainer = LLMForgeTrainer()
    trainer.train()


if __name__ == "__main__":
    train()