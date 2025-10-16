"""
Incremental Indexer
Tracks file changes and updates FAISS index incrementally without full rebuild
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
import pickle

from ansible_extractor import AnsibleExtractor, Chunk
from embedder import Embedder
from index_builder import IndexBuilder


class IncrementalIndexer:
    """Manage incremental updates to FAISS index"""
    
    def __init__(self, repo_path: str, index_path: str, 
                 embedder: Embedder, index_builder: IndexBuilder,
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize incremental indexer
        
        Args:
            repo_path: Path to repository
            index_path: Path to index directory
            embedder: Embedder instance
            index_builder: IndexBuilder instance
            chunk_size: Chunk size for extraction
            chunk_overlap: Overlap for smart chunking
        """
        self.repo_path = Path(repo_path)
        self.index_path = Path(index_path)
        self.embedder = embedder
        self.index_builder = index_builder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # State tracking
        self.state_file = self.index_path / 'indexer_state.json'
        self.file_hashes: Dict[str, str] = {}
        self.file_to_chunks: Dict[str, List[int]] = {}  # file -> chunk indices
        self.chunk_counter = 0
        
        self.load_state()
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def load_state(self):
        """Load indexer state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.file_hashes = state.get('file_hashes', {})
                self.file_to_chunks = state.get('file_to_chunks', {})
                self.chunk_counter = state.get('chunk_counter', 0)
                print(f"Loaded state: {len(self.file_hashes)} files tracked")
            except Exception as e:
                print(f"Warning: Could not load state: {e}")
                self.file_hashes = {}
                self.file_to_chunks = {}
                self.chunk_counter = 0
    
    def save_state(self):
        """Save indexer state to disk"""
        self.index_path.mkdir(parents=True, exist_ok=True)
        state = {
            'file_hashes': self.file_hashes,
            'file_to_chunks': self.file_to_chunks,
            'chunk_counter': self.chunk_counter,
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Saved state: {len(self.file_hashes)} files tracked")
    
    def scan_for_changes(self) -> Dict[str, List[Path]]:
        """
        Scan repository for changes
        
        Returns:
            Dict with 'added', 'modified', 'deleted' file lists
        """
        print("\nScanning repository for changes...")
        
        current_files: Set[str] = set()
        added_files: List[Path] = []
        modified_files: List[Path] = []
        
        # Scan current files
        extractor = AnsibleExtractor(
            repo_path=str(self.repo_path),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        for file_path in extractor._walk_repository():
            rel_path = str(file_path.relative_to(self.repo_path))
            current_files.add(rel_path)
            
            current_hash = self.compute_file_hash(file_path)
            
            if rel_path not in self.file_hashes:
                # New file
                added_files.append(file_path)
            elif self.file_hashes[rel_path] != current_hash:
                # Modified file
                modified_files.append(file_path)
        
        # Find deleted files
        deleted_files: List[str] = []
        for tracked_file in list(self.file_hashes.keys()):
            if tracked_file not in current_files:
                deleted_files.append(tracked_file)
        
        print(f"  Added: {len(added_files)} files")
        print(f"  Modified: {len(modified_files)} files")
        print(f"  Deleted: {len(deleted_files)} files")
        
        return {
            'added': added_files,
            'modified': modified_files,
            'deleted': deleted_files
        }
    
    def update_index(self, changes: Dict[str, List]) -> Dict[str, Any]:
        """
        Update index based on changes
        
        Args:
            changes: Dict with added, modified, deleted files
        
        Returns:
            Update statistics
        """
        stats = {
            'chunks_added': 0,
            'chunks_removed': 0,
            'files_processed': 0,
            'errors': []
        }
        
        # Process deleted files first
        for deleted_file in changes['deleted']:
            try:
                self.remove_file_from_index(deleted_file)
                stats['chunks_removed'] += len(self.file_to_chunks.get(deleted_file, []))
                del self.file_hashes[deleted_file]
                if deleted_file in self.file_to_chunks:
                    del self.file_to_chunks[deleted_file]
            except Exception as e:
                stats['errors'].append(f"Error removing {deleted_file}: {e}")
        
        # Process modified files (remove old, add new)
        for modified_file in changes['modified']:
            try:
                rel_path = str(modified_file.relative_to(self.repo_path))
                
                # Remove old chunks
                old_chunk_count = len(self.file_to_chunks.get(rel_path, []))
                self.remove_file_from_index(rel_path)
                stats['chunks_removed'] += old_chunk_count
                
                # Add new chunks
                new_chunks = self.add_file_to_index(modified_file)
                stats['chunks_added'] += len(new_chunks)
                stats['files_processed'] += 1
                
                # Update hash
                self.file_hashes[rel_path] = self.compute_file_hash(modified_file)
                
            except Exception as e:
                stats['errors'].append(f"Error updating {modified_file}: {e}")
        
        # Process added files
        for added_file in changes['added']:
            try:
                rel_path = str(added_file.relative_to(self.repo_path))
                
                # Add chunks
                new_chunks = self.add_file_to_index(added_file)
                stats['chunks_added'] += len(new_chunks)
                stats['files_processed'] += 1
                
                # Update hash
                self.file_hashes[rel_path] = self.compute_file_hash(added_file)
                
            except Exception as e:
                stats['errors'].append(f"Error adding {added_file}: {e}")
        
        return stats
    
    def add_file_to_index(self, file_path: Path) -> List[int]:
        """
        Add a file's chunks to the index
        
        Args:
            file_path: Path to file
        
        Returns:
            List of chunk indices added
        """
        rel_path = str(file_path.relative_to(self.repo_path))
        
        # Extract chunks from this file only
        extractor = AnsibleExtractor(
            repo_path=str(self.repo_path),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Extract content for this specific file
        chunks: List[Chunk] = []
        
        try:
            content = extractor._read_file(file_path)
            if not content:
                return []
            
            # Extract based on file type
            if file_path.suffix in {'.yml', '.yaml'}:
                # Try task-level extraction
                try:
                    import yaml
                    yaml_data = yaml.safe_load(content)
                    if yaml_data:
                        file_chunks = extractor._extract_tasks_from_yaml(yaml_data, file_path, content)
                        chunks.extend(file_chunks)
                except:
                    pass
            
            # If no task-level chunks, try smart or file-level
            if not chunks:
                line_count = len(content.split('\n'))
                if line_count > 300:
                    chunks = extractor._split_yaml_intelligently(content, file_path) if file_path.suffix in {'.yml', '.yaml'} else extractor._split_by_lines(content, file_path)
                else:
                    chunk = Chunk(
                        content=content,
                        metadata={
                            'file_path': rel_path,
                            'file_type': file_path.suffix,
                            'line_count': line_count,
                            'strategy': 'file_level'
                        },
                        chunk_type='file'
                    )
                    chunks = [chunk]
        
        except Exception as e:
            print(f"Warning: Error extracting chunks from {file_path}: {e}")
            return []
        
        if not chunks:
            return []
        
        # Generate embeddings
        chunks_dict = [c.to_dict() for c in chunks]
        embeddings, metadata = self.embedder.embed_chunks(chunks_dict, show_progress=False)
        
        # Add to FAISS index
        if self.index_builder.index is None:
            # Initialize index if needed
            self.index_builder.build_index(embeddings, metadata, index_type='flat')
            chunk_indices = list(range(len(embeddings)))
        else:
            # Add to existing index
            start_idx = self.index_builder.index.ntotal
            self.index_builder.index.add(embeddings.astype(np.float32))
            
            # Update metadata
            if self.index_builder.metadata is None:
                self.index_builder.metadata = []
            self.index_builder.metadata.extend(metadata)
            
            chunk_indices = list(range(start_idx, start_idx + len(embeddings)))
        
        # Track file to chunks mapping
        self.file_to_chunks[rel_path] = chunk_indices
        
        return chunk_indices
    
    def remove_file_from_index(self, rel_path: str):
        """
        Mark file's chunks as removed (FAISS doesn't support actual removal)
        
        Args:
            rel_path: Relative path to file
        """
        # Note: FAISS IndexFlatL2 doesn't support removing vectors
        # We mark them as invalid in metadata
        if rel_path in self.file_to_chunks:
            chunk_indices = self.file_to_chunks[rel_path]
            for idx in chunk_indices:
                if idx < len(self.index_builder.metadata):
                    self.index_builder.metadata[idx]['removed'] = True
    
    def rebuild_if_needed(self, removed_ratio_threshold: float = 0.3) -> bool:
        """
        Rebuild index if too many chunks are marked as removed
        
        Args:
            removed_ratio_threshold: Rebuild if removed chunks exceed this ratio
        
        Returns:
            True if rebuilt, False otherwise
        """
        if not self.index_builder.metadata:
            return False
        
        total_chunks = len(self.index_builder.metadata)
        removed_chunks = sum(1 for m in self.index_builder.metadata if m.get('removed', False))
        
        if removed_chunks / total_chunks > removed_ratio_threshold:
            print(f"\nRebuilding index: {removed_chunks}/{total_chunks} chunks removed ({removed_chunks/total_chunks*100:.1f}%)")
            self.rebuild_clean_index()
            return True
        
        return False
    
    def rebuild_clean_index(self):
        """Rebuild index without removed chunks"""
        # Filter out removed chunks
        valid_metadata = [m for m in self.index_builder.metadata if not m.get('removed', False)]
        
        if not valid_metadata:
            print("No valid chunks to rebuild")
            return
        
        # We need to re-generate embeddings or have them stored
        # For now, trigger a full reindex
        print("Full reindex required to clean removed chunks")
        print("Run: python3 rag_agent.py --build --force")


def main():
    """Test incremental indexer"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Incremental FAISS indexer')
    parser.add_argument('--repo', default=os.getenv('REPO_PATH', './RHEL8-CIS'), help='Repository path')
    parser.add_argument('--index-path', default=os.getenv('FAISS_INDEX_PATH', './faiss_index'), help='Index path')
    parser.add_argument('--scan', action='store_true', help='Scan for changes')
    parser.add_argument('--update', action='store_true', help='Update index with changes')
    
    args = parser.parse_args()
    
    # Load components
    print("Loading embedder and index...")
    embedder = Embedder()
    index_builder = IndexBuilder()
    
    try:
        index_builder.load_index(args.index_path)
    except FileNotFoundError:
        print("No existing index found. Run full build first.")
        return
    
    # Create incremental indexer
    indexer = IncrementalIndexer(
        repo_path=args.repo,
        index_path=args.index_path,
        embedder=embedder,
        index_builder=index_builder
    )
    
    if args.scan or args.update:
        changes = indexer.scan_for_changes()
        
        if args.update:
            print("\nUpdating index...")
            stats = indexer.update_index(changes)
            
            print("\nUpdate Statistics:")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Chunks added: {stats['chunks_added']}")
            print(f"  Chunks removed: {stats['chunks_removed']}")
            if stats['errors']:
                print(f"  Errors: {len(stats['errors'])}")
                for error in stats['errors'][:5]:
                    print(f"    - {error}")
            
            # Save updated index and state
            index_builder.save_index(args.index_path)
            indexer.save_state()
            
            # Check if rebuild needed
            indexer.rebuild_if_needed()
            
            print("\n✅ Incremental update complete!")


if __name__ == '__main__':
    main()

