"""
ChromaDB Vector Database Client
Store and retrieve document embeddings
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from config.ai_settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.utils.logger import setup_logger
import json

logger = setup_logger(__name__)


class ChromaDBClient:
    """ChromaDB client for vector storage and retrieval"""
    
    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME
    ):
        """
        Initialize ChromaDB client
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self._connect()
    
    def _connect(self):
        """Connect to ChromaDB"""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Desoutter repair manuals and bulletins"}
            )
            
            logger.info(f"✅ Connected to ChromaDB: {self.collection_name}")
            logger.info(f"   Documents in collection: {self.collection.count()}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            raise
    
    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        check_duplicates: bool = False
    ) -> int:
        """
        Add documents with embeddings to collection
        
        Args:
            chunks: List of chunk dicts with text and metadata
            embeddings: List of embedding vectors
            check_duplicates: If True, check specifically for content duplicates before adding
            
        Returns:
            Number of documents added
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        if not chunks:
            logger.warning("No chunks to add")
            return 0
            
        # Filter duplicates if requested
        if check_duplicates:
            try:
                # Get unique source filenames from chunks to narrow down search
                sources = set(c.get("metadata", {}).get("source") for c in chunks if c.get("metadata", {}).get("source"))
                
                # If all chunks are from same source(s), we can query efficiently
                existing_hashes = set()
                
                # Check duplication config
                from config.ai_settings import ENABLE_DEDUPLICATION
                if ENABLE_DEDUPLICATION:
                    for source in sources:
                        # Get existing chunks for this source
                        existing = self.collection.get(
                            where={"source": source},
                            include=["metadatas"]
                        )
                        
                        if existing and existing.get("metadatas"):
                            for meta in existing["metadatas"]:
                                if meta.get("content_hash"):
                                    existing_hashes.add(meta.get("content_hash"))
                    
                    if existing_hashes:
                        logger.info(f"Found {len(existing_hashes)} existing unique content hashes")
                        
                        # Filter chunks
                        unique_chunks = []
                        unique_embeddings = []
                        skipped_count = 0
                        
                        for i, chunk in enumerate(chunks):
                            content_hash = chunk.get("metadata", {}).get("content_hash")
                            if content_hash and content_hash in existing_hashes:
                                skipped_count += 1
                                continue
                            unique_chunks.append(chunk)
                            unique_embeddings.append(embeddings[i])
                            
                        if skipped_count > 0:
                            logger.info(f"Skipping {skipped_count} duplicate chunks")
                            chunks = unique_chunks
                            embeddings = unique_embeddings
                            
                        if not chunks:
                            logger.info("All chunks were duplicates. Nothing to add.")
                            return 0
            except Exception as e:
                logger.error(f"Error during deduplication check: {e}")
                # Fallback to adding all documents if deduplication check fails
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk["chunk_id"] for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            raw_metadatas = [chunk.get("metadata", {}) for chunk in chunks]

            # Sanitize metadata values: Chroma requires primitive types (str/int/float/bool)
            def sanitize_value(v):
                # If it's already a primitive acceptable type, return as-is
                if isinstance(v, (str, int, float, bool)):
                    return v
                # Lists/tuples -> comma-separated string
                if isinstance(v, (list, tuple)):
                    try:
                        return ", ".join(str(x) for x in v)
                    except Exception:
                        return str(v)
                # dicts or other objects -> JSON-like string
                try:
                    return str(v)
                except Exception:
                    return repr(v)

            metadatas = []
            for md in raw_metadatas:
                if not isinstance(md, dict):
                    # fallback to a simple dict with one field
                    metadatas.append({"meta": sanitize_value(md)})
                    continue

                sanitized = {}
                for k, val in md.items():
                    sanitized[k] = sanitize_value(val)
                metadatas.append(sanitized)

            # Debug: check for any remaining non-primitive values
            for i, md in enumerate(metadatas):
                for k, v in md.items():
                    if isinstance(v, (list, tuple, dict)):
                        logger.warning(f"Non-primitive metadata value remains at index {i} key '{k}': {type(v)} -> {v}")

            logger.debug(f"Sanitized metadatas sample: {metadatas[:2]}")

            # Debug: dump sanitized metadatas to logs
            try:
                logger.info(f"Sanitized metadatas (len={len(metadatas)}): {json.dumps(metadatas[:5], ensure_ascii=False)}")
            except Exception:
                logger.info(f"Sanitized metadatas sample (repr): {repr(metadatas[:5])}")

            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"✅ Added {len(chunks)} documents to ChromaDB")
            return len(chunks)

        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            raise
    
    def query(
        self,
        query_text: str,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Query similar documents
        
        Args:
            query_text: Query text (for logging)
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Query results
        """
        try:
            # Log the where filter to verify unsupported operators aren't used
            logger.info(f"Chroma query where filter: {where}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"Query: '{query_text[:50]}...' -> {len(results['ids'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            raise
    
    def get_count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
    
    def clear_collection(self):
        """Clear all documents from collection"""
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"✅ Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            raise
    
    def delete_by_source(self, source: str):
        """Delete all documents from a specific source"""
        try:
            self.collection.delete(where={"source": source})
            logger.info(f"✅ Deleted documents from source: {source}")
        except Exception as e:
            logger.error(f"❌ Error deleting documents: {e}")
            raise
