import os
import threading
import logging
import io
import contextlib
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Singleton manager for the ONNX embedding model (all-MiniLM-L6-v2).
    Shared across SemanticCache and QuestionDetector to prevent duplicate RAM overhead.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingManager, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self._model = None
        self._model_lock = threading.Lock()
        self._embed_fn = None
        self._model_name = "BAAI/bge-small-en-v1.5"

    def warmup(self) -> None:
        """Pre-load the model at app startup."""
        self._load_model()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            try:
                os.environ.setdefault("FASTEMBED_CACHE_PATH", "./data/cache/fastembed")
                # Silence HF Hub authentication noise (we only use public cached models)
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self._model_name)
                # fastembed returns generator
                self._embed_fn = lambda x: list(self._model.embed([x]))[0]
                
                # Check underlying cache directory to confirm quantization
                cache_dir = os.environ.get("FASTEMBED_CACHE_PATH", "./data/cache/fastembed")
                is_quantized = os.path.exists(os.path.join(cache_dir, "models--qdrant--bge-small-en-v1.5-onnx-q"))
                q_status = "INT8 Quantized" if is_quantized else "FP32"
                
                logger.info(f"✅ Shared EmbeddingManager loaded via fastembed ONNX ({self._model_name} | {q_status})")
                return
            except Exception as e:
                logger.debug(f"fastembed unavailable: {e}")

            # L-3: Two-stage SentenceTransformer fallback. The first attempt
            # uses the ONNX runtime backend (fast, no torch dep); only if that
            # specifically fails do we fall back to the default PyTorch
            # backend. These are NOT redundant: a user with sentence-
            # transformers installed but no onnxruntime will hit the second
            # branch. The first exception is logged at debug so the chain is
            # visible without spamming a warning on every cold start.
            try:
                from sentence_transformers import SentenceTransformer
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self._model = SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2",
                        backend="onnx",
                    )
                self._embed_fn = lambda x: self._model.encode(x)
                logger.info("✅ Shared EmbeddingManager loaded via sentence-transformers (ONNX backend)")
                return
            except Exception as e:
                logger.debug(f"sentence-transformers ONNX backend unavailable: {e}")

            try:
                from sentence_transformers import SentenceTransformer
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self._embed_fn = lambda x: self._model.encode(x)
                logger.info("✅ Shared EmbeddingManager loaded via sentence-transformers (PyTorch)")
                return
            except Exception as e:
                logger.warning(f"Shared EmbeddingManager failed to load: {e}")
                self._model = False
                self._embed_fn = lambda x: None

    def embed(self, text: str) -> np.ndarray | None:
        """Generate an L2-normalized embedding vector for a single string."""
        if not text:
            return None
        self._load_model()
        if not self._embed_fn or self._model is False:
            return None
        try:
            vec = self._embed_fn(text)
            norm = np.linalg.norm(vec)
            if norm > 0:
                return vec / norm
            return vec
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
            return None
