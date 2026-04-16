"""
RAGEngine - v4.1 (Layer 3 Hardened).
FIXED: Concurrency protection for initial vector DB loading.
RESTORATION: Implemented 'add_directory' with recursive walking and chunking.
"""

import os
import time
import threading
import asyncio
from pathlib import Path
from typing import List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGEngine:
    def __init__(self, config):
        self.config = config
        self.enabled = config.get("rag.enabled", True)
        self.top_k = config.get("rag.top_k", 5)
        self.chunk_size = config.get("rag.chunk_size", 512)
        self.chunk_overlap = config.get("rag.chunk_overlap", 50)
        
        self.collection = None
        self.client = None
        self._embed_fn = None
        self._loaded = False
        self._loading = False
        self._lock = threading.Lock()
        
        self._cache = {}
        self._cache_ttl = config.get("rag.cache_ttl", 60)

    def _ensure_loaded(self):
        """Hardened lazy-loader with concurrency protection."""
        if self._loaded: return

        with self._lock:
            if self._loaded or self._loading: return
            self._loading = True
            try:
                import chromadb
                from chromadb.config import Settings
                
                persist = self.config.get("rag.persist_dir", "./data/vectordb")
                os.makedirs(persist, exist_ok=True)

                # HARDENED: Handle potential handle collisions during restart
                try:
                    self.client = chromadb.PersistentClient(path=persist, settings=Settings(anonymized_telemetry=False))
                    self.collection = self.client.get_or_create_collection("knowledge", metadata={"hnsw:space": "cosine"})
                except Exception as ex:
                    logger.warning(f"RAG: Persistence handle sticky, retrying... ({ex})")
                    time.sleep(1.0)
                    self.client = chromadb.PersistentClient(path=persist, settings=Settings(anonymized_telemetry=False))
                    self.collection = self.client.get_or_create_collection("knowledge", metadata={"hnsw:space": "cosine"})

                # Initialize Embeddings (Fallback to SentenceTransformers if FastEmbed missing)
                try:
                    from fastembed import TextEmbedding
                    emb = TextEmbedding(model_name=self.config.get("rag.embedding_model", "BAAI/bge-small-en-v1.5"))
                    self._embed_fn = lambda texts: [e.tolist() for e in emb.embed(texts)]
                except:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    self._embed_fn = lambda texts: model.encode(texts).tolist()

                self._loaded = True
                logger.info(f"✅ RAG Ready ({self.collection.count()} chunks)")
            except Exception as e:
                logger.error(f"RAG Load Failure: {e}")
                self.enabled = False
            finally:
                self._loading = False

    async def query(self, text: str) -> List[str]:
        if not self.enabled: return []
        self._ensure_loaded()
        if not self.enabled or not self.collection: return []

        now = time.time()
        cache_key = text.strip().lower()
        if cache_key in self._cache and now < self._cache[cache_key][1]:
            return self._cache[cache_key][0]

        try:
            emb = self._embed_fn([text])
            count = self.collection.count()
            if count == 0: return []

            results = self.collection.query(
                query_embeddings=emb,
                n_results=min(self.top_k, count),
                include=["documents", "distances"],
            )

            if results and results["documents"]:
                docs = results["documents"][0] or []
                dists = results["distances"][0] if results["distances"] else []
                hits = [d.strip() for i, d in enumerate(docs) if d and (i >= len(dists) or dists[i] < 0.6)]
                final = hits[:self.top_k]
                self._cache[cache_key] = (final, now + self._cache_ttl)
                return final
        except Exception as e:
            logger.error(f"RAG Query Error: {e}")
        return []

    def add_directory(self, dir_path: str):
        """RESTORATION: Implemented recursive indexer for local knowledge."""
        self._ensure_loaded()
        if not self.enabled or not self.collection: return
        
        root = Path(dir_path)
        if not root.exists(): 
            os.makedirs(root, exist_ok=True)
            return

        logger.info(f"📚 Indexing directory: {dir_path}")
        supported = {'.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', '.yaml', '.yml', '.cpp', '.java'}
        
        try:
            documents = []
            metadatas = []
            ids = []
            
            for path in root.rglob('*'):
                if path.suffix in supported:
                    try:
                        content = path.read_text(encoding='utf-8', errors='ignore')
                        if not content.strip(): continue
                        
                        # Chunking
                        chunks = self._chunk_text(content, self.chunk_size, self.chunk_overlap)
                        for i, chunk in enumerate(chunks):
                            chunk_id = f"{path.name}_{i}_{int(time.time())}"
                            documents.append(chunk)
                            metadatas.append({"source": str(path), "chunk": i})
                            ids.append(chunk_id)
                    except Exception as e:
                        logger.warning(f"RAG: Skip file {path}: {e}")

            if documents:
                # Upsert into ChromaDB
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=self._embed_fn(documents)
                )
                logger.info(f"✅ RAG: Indexed {len(documents)} new chunks from {dir_path}")
        except Exception as e:
            logger.error(f"RAG: Directory Indexing Failed: {e}")

    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        """Simple sliding window chunker."""
        if len(text) <= size: return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks
