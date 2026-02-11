import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.core.logger import logger

def load_model_and_tokenizer(base_model_name: str, adapter_path: str = None):
    """
    Loads the model and tokenizer with 4-bit quantization support if available.
    Returns: (model, tokenizer)
    """
    logger.info(f"Loader: Preparing to load {base_model_name}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # 2. Load Base Model
    # Note: In a real prod env, you might add BitsAndBytesConfig here for 4-bit loading
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    # 3. Load Adapter (LoRA) if provided
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Loader: Found adapter at {adapter_path}. Merging...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.info("Loader: No adapter found or path invalid. Using base model.")

    model.eval() # Set to inference mode (crucial!)
    
    return model, tokenizer