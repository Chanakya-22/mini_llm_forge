import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(base_model_name: str, adapter_path: str):
    """
    Loads the quantized base model, resizes embeddings to match the fine-tuned
    tokenizer, and merges the LoRA adapter for inference.
    """
    logger.info(f"Loader: Preparing tokenizer from {adapter_path}...")
    # CRITICAL FIX: Load the tokenizer from the adapter path, because 
    # that is where our newly added <|im_start|> tokens are saved!
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Configure 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    logger.info(f"Loader: Preparing to load base model {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # CRITICAL FIX: Resize the base model's embeddings BEFORE applying the adapter
    # If the base model has 32000 tokens, but the tokenizer has 32002, we must resize.
    if len(tokenizer) != model.config.vocab_size:
        logger.info(f"Loader: Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
    logger.info(f"Loader: Found adapter at {adapter_path}. Merging...")
    # Now that the shapes match (32002 == 32002), we can safely load the adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Put the model in evaluation mode (turns off dropout layers, etc.)
    model.eval()
    
    return model, tokenizer