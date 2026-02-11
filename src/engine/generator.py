import torch
from src.core.config import settings
from src.core.logger import logger
from src.engine.loader import load_model_and_tokenizer

class LLMEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """
        Orchestrates the loading process using the loader module.
        """
        logger.info("Engine: Requesting model from loader...")
        
        self.model, self.tokenizer = load_model_and_tokenizer(
            base_model_name=settings.BASE_MODEL_NAME,
            adapter_path=settings.ADAPTER_PATH
        )
        
        logger.info("Engine: Model is ready for inference.")

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Singleton Instance
engine = LLMEngine()