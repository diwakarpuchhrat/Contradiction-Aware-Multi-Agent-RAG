import faiss
import numpy as np


class VectorStore:
    """
    Vector store using FAISS for similarity search.
    Automatically uses GPU-accelerated FAISS if available, otherwise falls back to CPU.
    """
    
    def __init__(self, dim):
        """
        Initialize vector store.
        
        Args:
            dim: Dimension of embeddings
        """
        self.index = self._create_index(dim)
        self.texts = []
        self.meta = []

    def _create_index(self, dim):
        """
        Create a FAISS index, preferring a GPU index when possible.
        """
        cpu_index = faiss.IndexFlatL2(dim)
        try:
            # If FAISS with GPU support is installed and GPUs are available, move index to GPU
            if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                return gpu_index
        except Exception:
            # Any issues with GPU setup will simply fall back to CPU index
            pass
        return cpu_index
    
    def add(self, embeddings, texts, metadata):
        """
        Add embeddings to the store.
        
        Args:
            embeddings: List or array of embeddings
            texts: List of text chunks corresponding to embeddings
            metadata: List of metadata dictionaries corresponding to embeddings
        """
        if len(embeddings) == 0:
            return
        
        embeddings_array = np.array(embeddings).astype("float32")
        self.index.add(embeddings_array)
        self.texts.extend(texts)
        self.meta.extend(metadata)
    
    def search(self, query_embedding, top_k=5):
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of dictionaries with 'text' and 'meta' keys
        """
        if self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "meta": self.meta[idx]
                })
        
        return results
    
    