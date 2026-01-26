from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config.experiment_config import (
    LLM_NAME,
    LLM_MAX_TOKENS,
)


class QwenLLM:
    def __init__(self, device: str = None, mock_mode: bool = False):
        self.mock_mode = mock_mode

        if self.mock_mode:
            print("Initializing QwenLLM in MOCK MODE (for testing)")
            self.device = "mock"
            self.tokenizer = None
            self.model = None
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing QwenLLM on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_NAME,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            self.model.to("cpu")

        self.model.eval()

    def generate(self, prompt: str) -> str:
        """
        Generate text from Qwen with SAFE, BOUNDED, TIME-EFFICIENT decoding.
        """

        if self.mock_mode:
            return (
                "This is a mock answer generated for testing purposes. "
                "The actual answer would appear here."
            )

        # 🔥 CRITICAL FIXES:
        # 1. Hard cap input tokens
        # 2. Disable padding for single-sample generation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,   # HARD INPUT CAP (most important)
            padding=False
        ).to(self.device)

        # Hard cap output tokens
        max_tokens = min(LLM_MAX_TOKENS, 128)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,  # speeds up decoding
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens
        generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

        decoded = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return decoded.strip()
