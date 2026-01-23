"""
EmbeddingsProvider abstraction for NirnAI RAG Review.
Supports swappable embedding backends (local SentenceTransformers or OpenAI).
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import os


class EmbeddingsProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerProvider(EmbeddingsProvider):
    """
    Local embedding provider using SentenceTransformers.
    Uses 'all-MiniLM-L6-v2' by default (384 dimensions, fast, free).
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingsProvider(EmbeddingsProvider):
    """
    OpenAI embedding provider stub.
    Plug in your OpenAI API key to use text-embedding-3-small or text-embedding-ada-002.
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Dimensions for common OpenAI embedding models
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = self._dimensions.get(model_name, 1536)
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string using OpenAI API."""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings using OpenAI API."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key to constructor."
            )
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )
        
        client = OpenAI(api_key=self.api_key)
        
        # OpenAI has a limit on batch size, process in chunks
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


def get_embeddings_provider(
    provider_type: str = "sentence_transformer",
    **kwargs
) -> EmbeddingsProvider:
    """
    Factory function to get an embeddings provider.
    
    Args:
        provider_type: One of "sentence_transformer" or "openai"
        **kwargs: Additional arguments passed to the provider constructor
    
    Returns:
        An EmbeddingsProvider instance
    
    Examples:
        >>> provider = get_embeddings_provider("sentence_transformer")
        >>> provider = get_embeddings_provider("openai", api_key="sk-...")
    """
    providers = {
        "sentence_transformer": SentenceTransformerProvider,
        "openai": OpenAIEmbeddingsProvider,
    }
    
    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Available: {list(providers.keys())}"
        )
    
    return providers[provider_type](**kwargs)
