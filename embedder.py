"""
Embedding Generator
Generates embeddings for text chunks using sentence-transformers
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None, use_pyvegas: bool = False):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_pyvegas: If True, use PyVegas VegasEmbeddingService for embeddings
        """
        self.backend = 'pyvegas' if use_pyvegas else 'sentence-transformers'
        self.model_name = model_name
        self.embedding_dim: Optional[int] = None
        self.model = None
        self.vegas = None
        
        if self.backend == 'pyvegas':
            try:
                # Lazy import to avoid hard dependency unless flag is used
                from pyvegas.langx import VegasEmbeddingService  # type: ignore
            except Exception as e:
                raise ImportError("PyVegas not installed. Please install 'pyvegas' to use --pyvegas mode.") from e
            print("Initializing PyVegas VegasEmbeddingService for embeddings...")
            self.vegas = VegasEmbeddingService()
            # Dimension will be inferred on first encode
            print("PyVegas embedding service initialized.")
        else:
            print(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 32, 
                    show_progress: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate embeddings for a list of chunks
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata'
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        if not chunks:
            return np.array([]), []
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text content
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        if self.backend == 'pyvegas':
            # VegasEmbeddingService exposes LangChain-compatible API
            vectors = self.vegas.embed_documents(texts)
            import numpy as np  # local import safe here
            embeddings = np.asarray(vectors, dtype=np.float32)
            # Normalize embeddings as done for sentence-transformers
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            if self.embedding_dim is None and embeddings.size:
                self.embedding_dim = embeddings.shape[1]
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
        
        # Prepare metadata (include chunk info)
        metadata_list = []
        for i, chunk in enumerate(chunks):
            meta = {
                'chunk_id': i,
                'content_preview': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                'content_length': len(chunk['content']),
                'metadata': chunk.get('metadata', {}),
                'chunk_type': chunk.get('chunk_type', 'unknown')
            }
            metadata_list.append(meta)
        
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings, metadata_list
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text
        
        Returns:
            Query embedding as numpy array
        """
        if self.backend == 'pyvegas':
            vector = self.vegas.embed_query(query)
            import numpy as np
            arr = np.asarray(vector, dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(arr) or 1.0
            arr = arr / norm
            if self.embedding_dim is None and arr.size:
                self.embedding_dim = arr.shape[0]
            return arr
        else:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding[0]
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], 
                       output_dir: str, filename_prefix: str = 'embeddings'):
        """
        Save embeddings and metadata to disk
        
        Args:
            embeddings: Embeddings array
            metadata: Metadata list
            output_dir: Output directory
            filename_prefix: Prefix for output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy array
        embeddings_file = output_path / f'{filename_prefix}.npy'
        np.save(embeddings_file, embeddings)
        print(f"Saved embeddings to {embeddings_file}")
        
        # Save metadata as JSON
        metadata_file = output_path / f'{filename_prefix}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'num_embeddings': len(embeddings),
                'metadata': metadata
            }, f, indent=2)
        print(f"Saved metadata to {metadata_file}")
    
    def load_embeddings(self, input_dir: str, filename_prefix: str = 'embeddings') -> Tuple[np.ndarray, List[Dict], Dict]:
        """
        Load embeddings and metadata from disk
        
        Args:
            input_dir: Input directory
            filename_prefix: Prefix for input files
        
        Returns:
            Tuple of (embeddings array, metadata list, info dict)
        """
        input_path = Path(input_dir)
        
        # Load embeddings
        embeddings_file = input_path / f'{filename_prefix}.npy'
        embeddings = np.load(embeddings_file)
        print(f"Loaded embeddings from {embeddings_file}")
        
        # Load metadata
        metadata_file = input_path / f'{filename_prefix}_metadata.json'
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        info = {
            'model_name': data.get('model_name', 'unknown'),
            'embedding_dim': data.get('embedding_dim', embeddings.shape[1]),
            'num_embeddings': data.get('num_embeddings', len(embeddings))
        }
        
        print(f"Loaded {len(metadata)} metadata entries")
        return embeddings, metadata, info


def main():
    """Test the embedder"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings for chunks')
    parser.add_argument('chunks_file', help='Path to chunks JSON file')
    parser.add_argument('--output-dir', '-o', default='embeddings', help='Output directory')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence-transformers model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Load chunks
    print(f"Loading chunks from {args.chunks_file}...")
    with open(args.chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Create embedder
    embedder = Embedder(model_name=args.model)
    
    # Generate embeddings
    embeddings, metadata = embedder.embed_chunks(chunks, batch_size=args.batch_size)
    
    # Save embeddings
    embedder.save_embeddings(embeddings, metadata, args.output_dir)
    
    print("\nEmbedding Statistics:")
    print(f"  Number of embeddings: {len(embeddings)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Total size: {embeddings.nbytes / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()

