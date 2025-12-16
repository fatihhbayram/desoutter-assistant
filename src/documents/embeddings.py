"""
Embeddings Generation Module
Generate vector embeddings for text chunks
"""
from typing import List
import torch  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from config.ai_settings import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingsGenerator:
    """Generate embeddings using sentence-transformers"""
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        batch_size: int = EMBEDDING_BATCH_SIZE
    ):
        """
        Initialize embeddings generator
        
        Args:
            model_name: Name of sentence-transformer model
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        
        logger.info(f"Initializing embeddings model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"✅ Model loaded on {self.device}")
            logger.info(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()
