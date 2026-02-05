import os
import time
import requests

from config.experiment_config import LLM_MAX_TOKENS


class APILLM:

    def __init__(
        self,
        model_name: str = "meta/llama-3.1-8b-instruct",
        timeout: int = 120,
        max_retries: int = 3,
        sleep_between_retries: float = 2.0,
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_between_retries = sleep_between_retries

        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY environment variable not set")

        self.endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return ""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(LLM_MAX_TOKENS),
        }

        for _ in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()

            except requests.exceptions.ReadTimeout:
                time.sleep(self.sleep_between_retries)

            except Exception:
                time.sleep(self.sleep_between_retries)

        return ""
