from typing import Dict, Any
import os
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()


class ModelService:

    def __init__(self):
        self.key = os.getenv("GEMINI_API_KEY")
        if not self.key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        self.modularize_prompt = os.getenv("MODULARIZE_PROMPT")
        if not self.modularize_prompt:
            raise EnvironmentError("MODULARIZE_PROMPT environment variable is not set.")
        self.test_prompt = os.getenv("TEST_PROMPT")
        if not self.test_prompt:
            raise EnvironmentError("TEST_PROMPT environment variable is not set.")
        self.client = genai.Client(api_key=self.key)

    def _handle_model(self, prompt):
        return self.client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=500),
            ),
        )

    def modularize(self, raw_code: str) -> Dict[str, Any]:

        start = time.perf_counter()
        full_prompt = f"{self.modularize_prompt} {raw_code}"
        response = self._handle_model(full_prompt)
        end = time.perf_counter()

        return {
            "time": f"{round((end - start) * 1000,2)} ms",
            "text": response.text,
            "tokens": {
                "prompt": response.usage_metadata.prompt_token_count,
                "thoughts": response.usage_metadata.thoughts_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count,
            },
        }

    def generate_test(self, raw_code: str) -> Dict[str, Any]:

        start = time.perf_counter()
        full_prompt = f"{self.test_prompt} {raw_code}"
        response = self._handle_model(full_prompt)
        end = time.perf_counter()

        return {
            "time": f"{round((end - start) * 1000,2)} ms",
            "text": response.text,
            "tokens": {
                "prompt": response.usage_metadata.prompt_token_count,
                "thoughts": response.usage_metadata.thoughts_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count,
            },
        }
