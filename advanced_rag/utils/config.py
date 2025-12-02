"""
Configuration management for Advanced RAG System.
"""

import os
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for the Advanced RAG system.
    
    Handles loading and managing configuration parameters.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional dictionary of configuration parameters
        """
        self.config = config_dict or {}
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            # Ingestion settings
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'min_chunk_size': 100,
            'chunk_method': 'fixed',
            
            # Indexing settings
            'embedding_dim': 384,
            'embedding_model': 'default',
            
            # Retrieval settings
            'top_k': 5,
            'similarity_threshold': 0.0,
            
            # Generation settings
            'model_name': 'default',
            'temperature': 0.7,
            'max_tokens': 500,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """
        Update multiple configuration values.
        
        Args:
            config_dict: Dictionary of configuration parameters
        """
        self.config.update(config_dict)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration parameters.
        
        Returns:
            Dictionary of all configuration parameters
        """
        return self.config.copy()
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Create configuration from environment variables.
        
        Returns:
            Config instance
        """
        config = cls()
        
        # Load configuration from environment variables if available
        env_mappings = {
            'RAG_CHUNK_SIZE': 'chunk_size',
            'RAG_CHUNK_OVERLAP': 'chunk_overlap',
            'RAG_TOP_K': 'top_k',
            'RAG_TEMPERATURE': 'temperature',
        }
        
        for env_key, config_key in env_mappings.items():
            value = os.environ.get(env_key)
            if value:
                # Convert to appropriate type
                if config_key in ['chunk_size', 'chunk_overlap', 'top_k']:
                    value = int(value)
                elif config_key in ['temperature']:
                    value = float(value)
                config.set(config_key, value)
        
        return config
