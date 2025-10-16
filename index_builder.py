"""
FAISS Index Builder
Builds and manages FAISS indices for similarity search
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from pathlib import Path


class IndexBuilder:
    """Build and manage FAISS indices"""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the index builder
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = None
    
    def build_index(self, embeddings: np.ndarray, metadata: List[Dict], 
                   index_type: str = 'flat') -> faiss.Index:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Embeddings array (N x D)
            metadata: List of metadata dicts
            index_type: Type of index ('flat', 'ivf', 'hnsw')
        
        Returns:
            FAISS index
        """
        print(f"Building FAISS index (type: {index_type})...")
        
        if embeddings.shape[0] != len(metadata):
            raise ValueError(f"Embeddings count ({embeddings.shape[0]}) doesn't match metadata count ({len(metadata)})")
        
        # Ensure embeddings are float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Create index based on type
        if index_type == 'flat':
            # Flat index - exact search, good for small to medium datasets
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == 'ivf':
            # IVF index - approximate search, good for large datasets
            nlist = min(100, embeddings.shape[0] // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            # Train the index
            print(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings)
        elif index_type == 'hnsw':
            # HNSW index - hierarchical navigable small world, good balance
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        print(f"Adding {len(embeddings)} embeddings to index...")
        index.add(embeddings)
        
        self.index = index
        self.metadata = metadata
        
        print(f"Index built successfully. Total vectors: {index.ntotal}")
        return index
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search the index for similar embeddings
        
        Args:
            query_embedding: Query embedding (D,)
            k: Number of results to return
        
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")
        
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get metadata for results
        result_metadata = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                result_metadata.append(self.metadata[idx])
            else:
                result_metadata.append({'error': 'Invalid index'})
        
        return distances[0], indices[0], result_metadata
    
    def save_index(self, output_dir: str, index_name: str = 'faiss_index'):
        """
        Save index and metadata to disk
        
        Args:
            output_dir: Output directory
            index_name: Name for the index files
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = output_path / f'{index_name}.bin'
        faiss.write_index(self.index, str(index_file))
        print(f"Saved FAISS index to {index_file}")
        
        # Save metadata
        metadata_file = output_path / f'{index_name}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'embedding_dim': self.embedding_dim,
                'num_vectors': self.index.ntotal,
                'index_type': self.index.__class__.__name__,
                'metadata': self.metadata
            }, f, indent=2)
        print(f"Saved metadata to {metadata_file}")
    
    def load_index(self, input_dir: str, index_name: str = 'faiss_index'):
        """
        Load index and metadata from disk
        
        Args:
            input_dir: Input directory
            index_name: Name of the index files
        """
        input_path = Path(input_dir)
        
        # Load FAISS index
        index_file = input_path / f'{index_name}.bin'
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        print(f"Loaded FAISS index from {index_file}")
        print(f"  Index type: {self.index.__class__.__name__}")
        print(f"  Total vectors: {self.index.ntotal}")
        
        # Load metadata
        metadata_file = input_path / f'{index_name}_metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        self.metadata = data['metadata']
        self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
        
        print(f"Loaded {len(self.metadata)} metadata entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        if self.index is None:
            return {'status': 'not_built'}
        
        return {
            'status': 'ready',
            'index_type': self.index.__class__.__name__,
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'metadata_count': len(self.metadata) if self.metadata else 0
        }


def main():
    """Test the index builder"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build FAISS index')
    parser.add_argument('embeddings_dir', help='Directory containing embeddings')
    parser.add_argument('--output-dir', '-o', default='faiss_index', help='Output directory')
    parser.add_argument('--index-type', choices=['flat', 'ivf', 'hnsw'], 
                       default='flat', help='Type of FAISS index')
    
    args = parser.parse_args()
    
    # Load embeddings
    embeddings_file = Path(args.embeddings_dir) / 'embeddings.npy'
    metadata_file = Path(args.embeddings_dir) / 'embeddings_metadata.json'
    
    print(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)
    
    print(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    metadata = data['metadata']
    embedding_dim = data['embedding_dim']
    
    print(f"Loaded {len(embeddings)} embeddings (dim: {embedding_dim})")
    
    # Build index
    builder = IndexBuilder(embedding_dim=embedding_dim)
    builder.build_index(embeddings, metadata, index_type=args.index_type)
    
    # Save index
    builder.save_index(args.output_dir)
    
    print("\nIndex Statistics:")
    stats = builder.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    print("\nTesting search with first embedding...")
    query = embeddings[0]
    distances, indices, result_meta = builder.search(query, k=3)
    
    print(f"Top 3 results:")
    for i, (dist, idx, meta) in enumerate(zip(distances, indices, result_meta)):
        print(f"  {i+1}. Distance: {dist:.4f}, Index: {idx}")
        print(f"     File: {meta.get('metadata', {}).get('file_path', 'unknown')}")


if __name__ == '__main__':
    main()

