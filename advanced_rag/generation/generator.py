"""
Generator for Advanced RAG System.

Handles response generation using retrieved context.
"""

from typing import List, Optional, Dict, Any
from .prompt_templates import PromptTemplates


class Generator:
    """
    Generator for producing responses with retrieved context.
    
    In a production system, this would integrate with actual LLM APIs
    (e.g., OpenAI, Anthropic, local models). This implementation provides
    a simple mock generator for demonstration purposes.
    """
    
    def __init__(
        self,
        model_name: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the generator.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_templates = PromptTemplates()
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response based on the prompt and optional context.
        
        Args:
            prompt: The input prompt or question
            context: Optional context to use for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response string
        """
        # In production, this would call an actual LLM
        # For demonstration, we return a mock response
        
        if context:
            full_prompt = self.prompt_templates.create_qa_prompt(
                context=context,
                question=prompt
            )
        else:
            full_prompt = prompt
        
        # Mock generation - in production, call actual LLM here
        response = self._mock_generate(full_prompt)
        
        return response
    
    def generate_with_contexts(
        self,
        question: str,
        contexts: List[str],
        separator: str = "\n\n"
    ) -> str:
        """
        Generate a response using multiple context pieces.
        
        Args:
            question: The question to answer
            contexts: List of context strings
            separator: Separator between contexts
            
        Returns:
            Generated response
        """
        combined_context = separator.join(contexts)
        return self.generate(prompt=question, context=combined_context)
    
    def generate_batch(
        self,
        prompts: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            contexts: Optional list of contexts (same length as prompts)
            
        Returns:
            List of generated responses
        """
        if contexts is None:
            contexts = [None] * len(prompts)
        
        responses = []
        for prompt, context in zip(prompts, contexts):
            response = self.generate(prompt=prompt, context=context)
            responses.append(response)
        
        return responses
    
    def _mock_generate(self, prompt: str) -> str:
        """
        Mock generation function for demonstration.
        
        In production, this would be replaced with actual LLM API calls.
        
        Args:
            prompt: Full prompt to generate from
            
        Returns:
            Mock response
        """
        # Simple mock response based on prompt length and content
        if "question" in prompt.lower() or "?" in prompt:
            return "Based on the provided context, here is the answer: [This is a mock response. In production, an actual LLM would generate the response based on the context and question.]"
        else:
            return "[Mock generated response. In production, this would be replaced with actual LLM-generated content.]"
    
    def set_temperature(self, temperature: float):
        """
        Set the generation temperature.
        
        Args:
            temperature: New temperature value (0-1)
        """
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int):
        """
        Set the maximum number of tokens to generate.
        
        Args:
            max_tokens: Maximum number of tokens
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
