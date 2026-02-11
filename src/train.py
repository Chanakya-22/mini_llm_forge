import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from src.core.config import settings
from src.core.logger import logger

def train():
    logger.info("Starting Training Pipeline...")
    
    # 1. Load Data
    dataset_file = "data/custom_data.jsonl"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Training data not found at {dataset_file}")
    
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    logger.info(f"Loaded {len(dataset)} training examples.")

    # 2. Config for QLoRA (Memory Efficient Training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 3. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for fp16 training

    # 4. LoRA Configuration (The Fine-Tuning Magic)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="model_output",
        num_train_epochs=3,          # How many times to loop over your data
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        learning_rate=2e-4,
        fp16=True,
        save_strategy="epoch",
        optim="paged_adamw_32bit",   # Saves memory
    )

    # 6. The Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    # 7. Train!
    logger.info("Training started... (This may take 10-30 minutes)")
    trainer.train()
    
    # 8. Save
    logger.info("Saving model to model_output/")
    trainer.model.save_pretrained("model_output")
    tokenizer.save_pretrained("model_output")
    logger.info("Training Complete!")

if __name__ == "__main__":
    train()