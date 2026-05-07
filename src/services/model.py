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
        self.base_prompt = os.getenv("PROMPT")
        if not self.base_prompt:
            raise EnvironmentError("PROMPT environment variable is not set.")
        self.client = genai.Client(api_key=self.key)

    def modularize(self, raw_code):

        start = time.perf_counter()
        full_prompt = f"{self.base_prompt} {raw_code}"
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=100)
            ),
        )
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
