import os
from typing import Optional

try:
    import google.generativeai as genai
    from google.api_core.client_options import ClientOptions
except ImportError:
    genai = None

class LLMOracle:
    def __init__(self, config: dict):
        self.mode = config.get('mode', 'mock')
        self.model = None

        raw_name = config.get('model_name', 'gemini-2.0-flash')
        self.model_name = f"models/{raw_name.removeprefix('models/')}"

        if self.mode == 'mock':
            return

        # ۳. بررسی پیش‌نیازها
        if genai is None:
            raise ImportError("Please install `google-generativeai` package.")

        api_key = os.getenv(config.get('api_key_env_var', 'METIS_API_KEY'))
        if not api_key:
            print(">> [Warning] API Key not found. LLM Oracle might fail.")
            return


        # ۴. کانفیگ و ساخت مدل
        genai.configure(
            api_key=api_key,
            transport='rest',
            client_options=ClientOptions(api_endpoint="https://api.metisai.ir")
        )
        self.model = genai.GenerativeModel(self.model_name)

    def get_abstractive_summary(self, full_text: str) -> str:
        """
        Generates a summary using Metis AI (Native Google SDK).
        """
        if self.mode == 'mock':
            print(">> [LLM Oracle] Running in MOCK mode (no API call).")
            return full_text[:500] + "..."
        
        if self.model is None:
            raise ValueError("Gemini Model is not initialized. Check your API Key.")


        try:
            # ارسال پرامپت برای خلاصه سازی
            prompt = f"Please provide a concise abstractive summary of the following text:\n\n{full_text}"
            
            response = self.model.generate_content(prompt)
            
            return response.text

        except Exception as e:
            print(f"Error calling Metis API: {e}")
            return ""
            
        #return "Taking care of houseplants is a rewarding hobby that requires patience and attention to detail. Using a high-quality potting mix is essential because regular garden soil is often too heavy for indoor use. Taking care of living things brings a sense of peace and improves the air quality in your living space."