import os
from typing import Optional

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

class LLMOracle:
    def __init__(self, config: dict):
        self.mode = config.get('mode', 'mock')
        self.model = config.get('model_name', 'facebook/bart-large-cnn')
        
        env_var_name = config.get('api_key_env_var', 'HF_TOKEN')
        self.api_key = os.getenv(env_var_name)
        
        self.client = None
        
        if self.mode != 'mock':
            if not self.api_key:
                print(f">> [Warning] {env_var_name} not found. LLM Oracle might fail.")
            else:
                self.client = InferenceClient(
                    provider="hf-inference",
                    api_key=self.api_key,
                )

    def get_abstractive_summary(self, full_text: str) -> str:
        """
        Generates a summary using Hugging Face Inference API.
        Uses the 'summarization' task specific method.
        """
        print(self.api_key)
        if self.mode == 'mock':
            print(">> [LLM Oracle] Running in MOCK mode (no API call).")
            return full_text[:500] + "..."
        
        # ۲. بررسی وجود کلاینت
        if self.client is None:
            raise ValueError("Hugging Face Client is not initialized. Check your HF_TOKEN.")

        try:
            truncated_text = full_text[:3500] 

            result = self.client.summarization(
                truncated_text,
                model=self.model
            )
            
            return result.summary_text

        except Exception as e:
            print(f"Error calling Hugging Face API: {e}")
            return ""