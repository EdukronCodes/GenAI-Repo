"""
LLM Generator Module
Generates responses using open-source language models
Supports both transformers library and Ollama
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional
import torch


class LLMGenerator:
    """Generates text using open-source language models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", use_ollama: bool = False, ollama_model: str = "llama2"):
        """
        Initialize LLM generator
        
        Args:
            model_name: HuggingFace model name for transformers
                Options:
                - "microsoft/DialoGPT-medium" (smaller, faster)
                - "gpt2" (very fast, lower quality)
                - "mistralai/Mistral-7B-Instruct-v0.2" (better quality, requires more memory)
            use_ollama: If True, use Ollama instead of transformers
            ollama_model: Ollama model name if use_ollama is True
        """
        self.use_ollama = use_ollama
        self.model_name = model_name
        
        if use_ollama:
            try:
                import ollama
                self.ollama_client = ollama
                self.ollama_model = ollama_model
                print(f"Using Ollama with model: {ollama_model}")
            except ImportError:
                print("Ollama not installed. Install with: pip install ollama")
                print("Falling back to transformers...")
                self.use_ollama = False
        
        if not self.use_ollama:
            print(f"Loading model: {model_name}")
            print("This may take a few minutes on first run...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("Model loaded successfully")
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated text
        """
        if self.use_ollama:
            return self._generate_ollama(prompt, max_length, temperature)
        else:
            return self._generate_transformers(prompt, max_length, temperature)
    
    def _generate_ollama(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate using Ollama"""
        response = self.ollama_client.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={
                'num_predict': max_length,
                'temperature': temperature
            }
        )
        return response['response']
    
    def _generate_transformers(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate using transformers library"""
        outputs = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = outputs[0]['generated_text']
        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def generate_with_context(self, query: str, context: List[str], max_length: int = 512) -> str:
        """
        Generate response with retrieved context
        
        Args:
            query: User query
            context: List of retrieved context texts
            max_length: Maximum length of generated response
            
        Returns:
            Generated response
        """
        # Build prompt with context
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""Based on the following context, please answer the question.

{context_text}

Question: {query}

Answer:"""
        
        return self.generate(prompt, max_length=max_length, temperature=0.7)
