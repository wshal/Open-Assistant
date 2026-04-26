import unittest
import os
import glob
import time
from ai.embedding_manager import EmbeddingManager

class TestB3Int8Quantization(unittest.TestCase):
    """
    Q22/B3: Verifies that the Semantic Cache embedding manager is correctly utilizing
    INT8 ONNX Quantization to minimize VRAM/RAM overhead and maximize inference speed.
    """

    def setUp(self):
        self.em = EmbeddingManager()
        self.em.warmup()
        
    def test_int8_quantization_active(self):
        """B3: Verify that the loaded ONNX model is the quantized version (< 70MB)."""
        # Fastembed downloads the model to FASTEMBED_CACHE_PATH
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH", "./data/cache/fastembed")
        self.assertTrue(os.path.exists(cache_dir), f"Cache dir {cache_dir} not found")
        
        # The BAAI/bge-small-en-v1.5 model is mapped to qdrant/bge-small-en-v1.5-onnx-q
        # The 'onnx-q' suffix explicitly denotes the INT8 Quantized repository.
        repo_dir = os.path.join(cache_dir, "models--qdrant--bge-small-en-v1.5-onnx-q")
        self.assertTrue(os.path.exists(repo_dir), "Quantized ONNX model repository not found. The model is NOT using INT8!")
        
        # Find the actual .onnx file to verify its size.
        # FP32 is ~133MB. INT8 is ~67MB.
        onnx_files = glob.glob(os.path.join(repo_dir, "**", "*.onnx"), recursive=True)
        self.assertTrue(len(onnx_files) > 0, "No .onnx file found in the quantized repository")
        
        for onnx_file in onnx_files:
            size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
            # The INT8 model is typically around 67MB. We assert it's strictly under 75MB.
            self.assertLess(size_mb, 75.0, f"Model file {onnx_file} is {size_mb:.1f}MB, which indicates FP32. Expected INT8 (< 75MB).")
            print(f"\n[B3 INT8 Verification] Found quantized model: {os.path.basename(onnx_file)} ({size_mb:.1f} MB)")

    def test_embedding_latency(self):
        """Verify that the INT8 Quantized model performs inference within acceptable bounds (< 50ms)."""
        # Warmup query
        self.em.embed("Warmup query to load ONNX session into memory.")
        
        start = time.time()
        vec = self.em.embed("What is the computational complexity of INT8 quantized embedding inference?")
        duration_ms = (time.time() - start) * 1000
        
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 384)  # bge-small output dimension
        
        print(f"\n[B3 INT8 Latency] Single vector embedding generated in {duration_ms:.1f}ms")
        # INT8 ONNX should comfortably run under 50ms on most modern CPUs.
        self.assertLess(duration_ms, 50.0, f"Embedding took {duration_ms:.1f}ms, which is slower than expected for INT8.")

if __name__ == "__main__":
    unittest.main()
