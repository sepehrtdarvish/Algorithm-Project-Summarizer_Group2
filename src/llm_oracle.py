import os
import json
import re
from typing import Optional, Dict

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

        if genai is None:
            raise ImportError("Please install `google-generativeai` package.")

        api_key = os.getenv(config.get('api_key_env_var', 'METIS_API_KEY'))
        if not api_key:
            print(">> [Warning] API Key not found. LLM Oracle might fail.")
            return

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
            return full_text[:200] + "..."
        
        if self.model is None:
            raise ValueError("Gemini Model is not initialized. Check your API Key.")

        try:
            prompt = f"Please provide a concise abstractive summary of the following text:\n\n{full_text}"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Metis API: {e}")
            return ""

    def evaluate_batch_summaries(self, original_text: str, summaries: Dict[str, str]) -> Dict[str, int]:
        """
        Evaluates multiple summaries in a SINGLE prompt (Batch Evaluation).
        Returns a dictionary of scores {method_name: score_0_to_100}.
        """
        if self.mode == 'mock':
            import random
            return {k: random.randint(30, 90) for k in summaries.keys()}

        if self.model is None:
             raise ValueError("Gemini Model is not initialized.")

        # ساخت پرامپت دسته‌ای
        candidates_text = ""
        for name, summary in summaries.items():
            # تمیزکاری متن خلاصه (حذف خطوط خالی اضافه)
            clean_summary = summary.replace("\n", " ").strip()
            candidates_text += f"\n--- CANDIDATE ID: [{name}] ---\n{clean_summary}\n"

        prompt = f"""
        You are an expert judge of text summarization quality.
        I will provide an ORIGINAL TEXT and several CANDIDATE SUMMARIES.

        Your task:
        1. Read the original text carefully.
        2. Evaluate each candidate summary based on Coverage, Accuracy, Conciseness, and Coherence.
        3. Assign a score from 0 to 100 to EACH candidate.
        4. Compare them against each other to ensure fair relative ranking.

        ORIGINAL TEXT:
        "{original_text}"

        {candidates_text}

        --------------------------------------------------
        OUTPUT FORMAT:
        You MUST return ONLY a valid JSON object. Keys are Candidate IDs, values are Integer scores.
        Example:
        {{
            "TextRank": 45,
            "Frequency": 50,
            "Pure LLM": 85
        }}
        Do NOT write markdown code blocks. Just the JSON string.
        """

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            
            # پاکسازی خروجی برای استخراج JSON خالص با Regex
            # گاهی مدل توضیحات اضافه می‌دهد، ما فقط قسمت {...} را می‌خواهیم
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_text.replace("```json", "").replace("```", "")
            
            scores = json.loads(json_str)
            return scores

        except Exception as e:
            print(f">> [Error] LLM Evaluation failed: {e}")
            # در صورت خطا، نمره صفر برمی‌گردانیم تا برنامه متوقف نشود
            return {k: 0 for k in summaries.keys()}