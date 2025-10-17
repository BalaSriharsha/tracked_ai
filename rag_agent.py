"""
RAG-Based Ansible Agent
Main entry point for building index and querying the Ansible repository
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Any, List
from dotenv import load_dotenv

# Import our modules
from ansible_extractor import AnsibleExtractor
from embedder import Embedder
from index_builder import IndexBuilder
from rag_query import RAGQueryEngine


class RAGAgent:
    """Main RAG Agent for Ansible repositories"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG Agent
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load environment variables
        load_dotenv()
        
        # Set configuration with defaults
        self.repo_path = self.config.get('repo_path', os.getenv('REPO_PATH', './RHEL8-CIS'))
        self.index_path = self.config.get('index_path', os.getenv('FAISS_INDEX_PATH', './faiss_index'))
        self.embedding_model = self.config.get('embedding_model', 
                                               os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
        self.chunk_size = int(self.config.get('chunk_size', os.getenv('CHUNK_SIZE', '1000')))
        self.chunk_overlap = int(self.config.get('chunk_overlap', os.getenv('CHUNK_OVERLAP', '200')))
        self.top_k = int(self.config.get('top_k', os.getenv('TOP_K_RESULTS', '5')))
        
        # Initialize components
        self.extractor = None
        self.embedder = None
        self.index_builder = None
        self.query_engine = None
        self.use_pyvegas = bool(self.config.get('use_pyvegas', False))
    
    def build_index(self, force_rebuild: bool = False):
        """
        Build the FAISS index from the Ansible repository
        
        Args:
            force_rebuild: Force rebuild even if index exists
        """
        index_file = Path(self.index_path) / 'faiss_index.bin'
        
        if index_file.exists() and not force_rebuild:
            print(f"Index already exists at {self.index_path}")
            print("Use --force to rebuild")
            return
        
        print("="*80)
        print("BUILDING RAG INDEX")
        print("="*80)
        print(f"Repository: {self.repo_path}")
        print(f"Index path: {self.index_path}")
        print(f"Embedding model: {self.embedding_model}")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        print("="*80 + "\n")
        
        # Step 1: Extract content
        print("\n" + "="*80)
        print("STEP 1: EXTRACTING CONTENT")
        print("="*80)
        self.extractor = AnsibleExtractor(
            repo_path=self.repo_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = self.extractor.extract_all()
        
        if not chunks:
            print("ERROR: No chunks extracted from repository!")
            return
        
        # Convert chunks to dict format
        chunks_dict = [chunk.to_dict() for chunk in chunks]
        
        # Step 2: Generate embeddings
        print("\n" + "="*80)
        print("STEP 2: GENERATING EMBEDDINGS")
        print("="*80)
        self.embedder = Embedder(model_name=self.embedding_model, use_pyvegas=self.use_pyvegas)
        embeddings, metadata = self.embedder.embed_chunks(chunks_dict)
        
        # Step 3: Build FAISS index
        print("\n" + "="*80)
        print("STEP 3: BUILDING FAISS INDEX")
        print("="*80)
        self.index_builder = IndexBuilder(embedding_dim=embeddings.shape[1])
        self.index_builder.build_index(embeddings, metadata, index_type='flat')
        
        # Step 4: Save index
        print("\n" + "="*80)
        print("STEP 4: SAVING INDEX")
        print("="*80)
        self.index_builder.save_index(self.index_path, index_name='faiss_index')
        
        print("\n" + "="*80)
        print("INDEX BUILD COMPLETE!")
        print("="*80)
        print(f"Total chunks: {len(chunks)}")
        print(f"Total embeddings: {len(embeddings)}")
        print(f"Index saved to: {self.index_path}")
        print("="*80 + "\n")
    
    def load_index(self):
        """Load existing FAISS index"""
        print(f"Loading index from {self.index_path}...")
        
        # Load embedder
        self.embedder = Embedder(model_name=self.embedding_model, use_pyvegas=self.use_pyvegas)
        
        # Load index
        self.index_builder = IndexBuilder()
        self.index_builder.load_index(self.index_path, index_name='faiss_index')
        
        # Create query engine
        self.query_engine = RAGQueryEngine(
            embedder=self.embedder,
            index_builder=self.index_builder
        )
        
        print("Index loaded successfully!\n")
    
    def query(self, question: str):
        """
        Query the RAG system
        
        Args:
            question: User question
        """
        if self.query_engine is None:
            self.load_index()
        
        result = self.query_engine.query(question, top_k=self.top_k)
        self.query_engine.print_result(result)
        
        return result
    
    def interactive_mode(self):
        """Run the agent in interactive mode"""
        if self.query_engine is None:
            self.load_index()
        
        print("\n" + "="*80)
        print("RAG-BASED ANSIBLE AGENT - INTERACTIVE MODE")
        print("="*80)
        print(f"Repository: {self.repo_path}")
        print(f"Embedding Model: {self.embedding_model}")
        print(f"Top-K Results: {self.top_k}")
        print("="*80)
        print("\nType your questions about the Ansible repository")
        print("Commands: 'quit' or 'exit' to quit, 'stats' for index statistics, 'modify' to change code, 'update' to update index, 'help' for help\n")
        
        while True:
            try:
                question = input("\n>>> ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() in ['help', 'h', '?']:
                    print("\nAvailable commands:")
                    print("  - Ask any question to query the index")
                    print("  - stats           Show index statistics")
                    print("  - modify          Start interactive code modification with approvals")
                    print("  - update          Incrementally update the index")
                    print("  - quit/exit/q     Exit")
                    continue
                
                if question.lower() == 'stats':
                    self.print_stats()
                    continue
                
                if question.lower().startswith('update'):
                    self.update_index()
                    continue
                
                # Interactive modify flow
                if question.lower().startswith('modify'):
                    # Parse inline instruction if provided: e.g. "modify Update SSH settings"
                    parts = question.split(' ', 1)
                    if len(parts) > 1 and parts[1].strip():
                        instruction = parts[1].strip()
                    else:
                        instruction = input("Instruction: ").strip()
                        if not instruction:
                            print("No instruction provided. Cancelling.")
                            continue
                    
                    files_input = input("Target files (comma-separated, blank to auto-discover): ").strip()
                    target_files = None
                    if files_input:
                        target_files = [f.strip() for f in files_input.split(',') if f.strip()]
                        if not target_files:
                            target_files = None
                    
                    dry_resp = input("Dry run? (y/N): ").strip().lower()
                    dry_run = dry_resp in ['y', 'yes']
                    
                    mode_resp = input("Approval mode [i]nteractive/[a]uto (default i): ").strip().lower()
                    auto_approve = mode_resp in ['a', 'auto']
                    
                    stats = self.modify_code(
                        instruction=instruction,
                        target_files=target_files,
                        auto_approve=auto_approve,
                        dry_run=dry_run,
                        require_final_confirm=True
                    )
                    
                    # Offer to update index if changes applied and not a dry run
                    if isinstance(stats, dict) and stats.get('applied', 0) > 0 and not dry_run:
                        upd = input("Update index now to reflect changes? (y/N): ").strip().lower()
                        if upd in ['y', 'yes']:
                            self.update_index()
                    
                    continue
                
                if not question:
                    continue
                
                # Query the system
                result = self.query(question)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
    
    def print_stats(self):
        """Print index statistics"""
        if self.index_builder is None:
            print("No index loaded")
            return
        
        stats = self.index_builder.get_stats()
        print("\n" + "="*80)
        print("INDEX STATISTICS")
        print("="*80)
        for key, value in stats.items():
            print(f"{key:20s}: {value}")
        print("="*80)
    
    def update_index(self):
        """Incrementally update the FAISS index with changes"""
        from incremental_indexer import IncrementalIndexer
        
        print("\n" + "="*80)
        print("INCREMENTAL INDEX UPDATE")
        print("="*80 + "\n")
        
        # Load existing index and embedder
        if self.embedder is None:
            self.embedder = Embedder(model_name=self.embedding_model)
        
        if self.index_builder is None:
            self.index_builder = IndexBuilder()
            try:
                self.index_builder.load_index(self.index_path, index_name='faiss_index')
            except FileNotFoundError:
                print("❌ No existing index found. Run --build first.")
                return
        
        # Create incremental indexer
        indexer = IncrementalIndexer(
            repo_path=str(self.repo_path),
            index_path=str(self.index_path),
            embedder=self.embedder,
            index_builder=self.index_builder,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Scan for changes
        changes = indexer.scan_for_changes()
        
        total_changes = len(changes['added']) + len(changes['modified']) + len(changes['deleted'])
        
        if total_changes == 0:
            print("\n✅ No changes detected. Index is up to date.")
            return
        
        # Update index
        print("\nUpdating index...")
        stats = indexer.update_index(changes)
        
        # Display statistics
        print("\n" + "="*80)
        print("UPDATE STATISTICS")
        print("="*80)
        print(f"Files processed: {stats['files_processed']}")
        print(f"Chunks added: {stats['chunks_added']}")
        print(f"Chunks removed: {stats['chunks_removed']}")
        if stats['errors']:
            print(f"Errors: {len(stats['errors'])}")
            for error in stats['errors'][:3]:
                print(f"  - {error}")
        print("="*80 + "\n")
        
        # Save updated index and state
        self.index_builder.save_index(self.index_path, index_name='faiss_index')
        indexer.save_state()
        
        # Check if full rebuild needed
        if indexer.rebuild_if_needed():
            print("⚠️  Consider running a full rebuild soon")
        
        print("\n✅ Incremental update complete!")
    
    def modify_code(self, instruction: str, target_files: Optional[List[str]] = None,
                   auto_approve: bool = False, dry_run: bool = False, require_final_confirm: bool = False):
        """
        Modify code with approval workflow
        
        Args:
            instruction: What to modify
            target_files: Specific files to modify
            auto_approve: Auto-approve all changes
            dry_run: Show changes without applying
            require_final_confirm: Ask for final confirmation before applying (used in interactive mode)
        """
        from code_modifier import CodeModifierAgent
        
        # Load query engine if not loaded (for finding relevant files)
        if self.query_engine is None and target_files is None:
            try:
                self.load_index()
            except:
                print("⚠️  Could not load index. Specify --files explicitly.")
                return {'applied': 0, 'errors': 0}
        
        # Create code modifier
        modifier = CodeModifierAgent(
            repo_path=str(self.repo_path),
            rag_query_engine=self.query_engine
        )
        
        # Plan modifications
        modifications = modifier.plan_modification(instruction, target_files=target_files)
        
        if not modifications:
            print("\n❌ No modifications planned")
            return {'applied': 0, 'errors': 0}
        
        # Review
        modifier.review_modifications()
        
        # Approve
        if auto_approve:
            modifier.approve_all()
        else:
            modifier.interactive_approval()
            
        # Final confirmation before apply (only when not auto-approving)
        if require_final_confirm and not auto_approve:
            approved_paths = [m.file_path for m in modifier.modifications if m.approved]
            approved_count = len(approved_paths)
            if approved_count == 0:
                print("\nNo approved modifications to apply")
                return {'applied': 0, 'errors': 0}
            print("\n" + "="*80)
            print("FINAL APPROVAL")
            print("="*80)
            print(f"About to apply {approved_count} approved modification(s):")
            for p in approved_paths[:10]:
                print(f"  - {p}")
            if approved_count > 10:
                print(f"  ... and {approved_count - 10} more")
            resp = input("Proceed to apply changes? (y/N): ").strip().lower()
            if resp not in ['y', 'yes']:
                print("\n❌ Application cancelled")
                return {'applied': 0, 'errors': 0}
        
        # Apply
        stats = modifier.apply_modifications(dry_run=dry_run)
        
        if stats['applied'] > 0 and not dry_run:
            print("\n" + "="*80)
            print("RECOMMENDATION")
            print("="*80)
            print("Modified files detected. Consider updating the index:")
            print(f"  python3 rag_agent.py --update")
            print("="*80 + "\n")
        
        return stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RAG-Based Ansible Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from repository
  python3 rag_agent.py --build --repo ./RHEL8-CIS
  
  # Incrementally update index with changes
  python3 rag_agent.py --update
  
  # Query the index
  python3 rag_agent.py --query "How do I configure firewall rules?"
  
  # Interactive mode
  python3 rag_agent.py --interactive
  
  # Modify code with approval workflow
  python3 rag_agent.py --modify "Add a comment to explain the firewall rule"
  
  # Modify specific files (dry run)
  python3 rag_agent.py --modify "Update SSH settings" --files tasks/ssh.yml --dry-run
  
  # Modify with auto-approval
  python3 rag_agent.py --modify "Fix typos in comments" --auto-approve
  
  # Force rebuild index
  python3 rag_agent.py --build --force

Note: Use 'python3' instead of 'python' (Python 3.8+ required)
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--build', action='store_true', help='Build FAISS index from repository')
    mode_group.add_argument('--update', action='store_true', help='Incrementally update index with changes')
    mode_group.add_argument('--query', '-q', type=str, help='Query the index with a question')
    mode_group.add_argument('--interactive', '-i', action='store_true', help='Interactive query mode')
    mode_group.add_argument('--modify', type=str, help='Modify code with approval workflow')
    
    # Configuration options
    parser.add_argument('--repo', default=None, help='Path to Ansible repository')
    parser.add_argument('--index-path', default=None, help='Path to FAISS index directory')
    parser.add_argument('--model', default=None, help='Sentence-transformers model name')
    parser.add_argument('--chunk-size', type=int, default=None, help='Chunk size for smart chunking')
    parser.add_argument('--chunk-overlap', type=int, default=None, help='Chunk overlap')
    parser.add_argument('--top-k', type=int, default=None, help='Number of results to retrieve')
    parser.add_argument('--force', action='store_true', help='Force rebuild index')
    parser.add_argument('--pyvegas', action='store_true', help='Use PyVegas VegasEmbeddingService for embeddings')
    
    # Code modification options
    parser.add_argument('--files', nargs='+', help='Specific files to modify (for --modify mode)')
    parser.add_argument('--auto-approve', action='store_true', help='Auto-approve all modifications')
    parser.add_argument('--dry-run', action='store_true', help='Dry run - show changes without applying')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {}
    if args.repo:
        config['repo_path'] = args.repo
    if args.index_path:
        config['index_path'] = args.index_path
    if args.model:
        config['embedding_model'] = args.model
    if args.chunk_size:
        config['chunk_size'] = args.chunk_size
    if args.chunk_overlap:
        config['chunk_overlap'] = args.chunk_overlap
    if args.top_k:
        config['top_k'] = args.top_k
    if args.pyvegas:
        config['use_pyvegas'] = True
    
    # Create agent
    agent = RAGAgent(config=config)
    
    # Execute mode
    try:
        if args.build:
            agent.build_index(force_rebuild=args.force)
        elif args.update:
            agent.update_index()
        elif args.query:
            agent.query(args.query)
        elif args.interactive:
            agent.interactive_mode()
        elif args.modify:
            agent.modify_code(
                instruction=args.modify,
                target_files=args.files,
                auto_approve=args.auto_approve,
                dry_run=args.dry_run
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

