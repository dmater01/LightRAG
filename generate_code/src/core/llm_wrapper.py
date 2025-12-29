import os
import yaml
import torch
import logging
import numpy as np
from typing import Optional, List, Union

# OpenAI imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# HuggingFace imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.utils.logger import setup_logger

class LLMWrapper:
    """
    Unified interface for OpenAI API and Local HuggingFace models.
    Handles initialization, generation, and perplexity calculation.
    """
    
    def __init__(self, model_type: str = "local", config_path: str = "config/models.yaml"):
        """
        Initialize the LLM Wrapper.
        
        Args:
            model_type: "openai" or "local"
            config_path: Path to the models configuration file
        """
        self.logger = setup_logger(f"llm_wrapper_{model_type}")
        self.model_type = model_type.lower()
        self.config = self._load_config(config_path)
        
        self.model = None
        self.tokenizer = None
        self.client = None
        
        if self.model_type == "openai":
            self._setup_openai()
        elif self.model_type == "local":
            self._setup_local()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'openai' or 'local'.")

    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        if self.model_type == "openai":
            return full_config['models']['openai']
        else:
            return full_config['models']['local']

    def _setup_openai(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed.")
        
        api_key_env = self.config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            self.logger.warning(f"Environment variable {api_key_env} not found. OpenAI calls may fail.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = self.config.get('default_model', 'gpt-3.5-turbo-instruct')
        self.temperature = self.config.get('temperature', 0.1)
        self.logger.info(f"Initialized OpenAI client with model {self.model_name}")

    def _setup_local(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers or torch package is not installed.")
        
        self.model_name = self.config.get('default_model')
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = self.config.get('temperature', 0.1)
        self.max_new_tokens = self.config.get('max_new_tokens', 100)
        
        self.logger.info(f"Loading local model {self.model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Ensure pad token exists for batch processing if needed, though we mostly do single
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cpu":
                self.model.to(self.device)
                
            self.model.eval()
            self.logger.info("Local model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load local model: {e}")
            raise e

    def query(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: Input text
            max_tokens: Optional override for max generation length
            
        Returns:
            Generated text string
        """
        if self.model_type == "openai":
            return self._query_openai(prompt, max_tokens)
        else:
            return self._query_local(prompt, max_tokens)

    def _query_openai(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            # Handle chat models vs completion models if needed
            # Assuming completion model based on 'gpt-3.5-turbo-instruct' default
            # If chat model (gpt-4, gpt-3.5-turbo), need chat completions
            
            is_chat = "turbo" in self.model_name and "instruct" not in self.model_name or "gpt-4" in self.model_name
            
            if is_chat:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens if max_tokens else 100 # Default for safety
                )
                return response.choices[0].message.content.strip()
            else:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=max_tokens if max_tokens else 100
                )
                return response.choices[0].text.strip()
                
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"

    def _query_local(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            limit = max_tokens if max_tokens is not None else self.max_new_tokens
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=limit,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Local inference error: {e}")
            return f"Error: {str(e)}"

    def get_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of the text using the local model.
        Used for PPL Detection defense.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score (float)
        """
        if self.model_type != "local":
            # If we are in OpenAI mode, we can't easily calc PPL. 
            # The defense mechanism should instantiate a separate local LLMWrapper for PPL calculation.
            raise NotImplementedError("Perplexity calculation is only supported for local models.")
            
        try:
            encodings = self.tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                
            return torch.exp(loss).item()
            
        except Exception as e:
            self.logger.error(f"Perplexity calculation error: {e}")
            return float('inf')

    def get_windowed_perplexity(self, text: str, window_size: int = 50) -> float:
        """
        Calculate max perplexity over sliding windows.
        
        Args:
            text: Input text
            window_size: Size of the sliding window in tokens
            
        Returns:
            Max perplexity score
        """
        if self.model_type != "local":
            raise NotImplementedError("Windowed PPL is only supported for local models.")
            
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        
        if seq_len < window_size:
            return self.get_perplexity(text)
            
        nlls = []
        stride = window_size // 2 # Overlap by half
        
        # Sliding window implementation
        # We'll compute PPL for each window
        # Note: The paper says "Max PPL of sliding windows".
        
        max_ppl = 0.0
        
        for i in range(0, seq_len, stride):
            begin_loc = i
            end_loc = min(i + window_size, seq_len)
            trg_len = end_loc - i  # may be less than window_size at the end
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                # loss is the average NLL of the window
                neg_log_likelihood = outputs.loss
            
            ppl = torch.exp(neg_log_likelihood).item()
            if ppl > max_ppl:
                max_ppl = ppl
                
            if end_loc == seq_len:
                break
                
        return max_ppl
