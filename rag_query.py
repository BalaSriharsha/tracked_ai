"""
RAG Query Engine
Retrieves relevant chunks and generates answers using LLM
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# from pyvegas.langx.llm import VegasChatLLM

@dataclass
class QueryResult:
    """Result from a RAG query"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    context_used: str


class RAGQueryEngine:
    """Query engine for RAG-based question answering"""
    
    def __init__(self, embedder, index_builder, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the query engine
        
        Args:
            embedder: Embedder instance
            index_builder: IndexBuilder instance
            llm_config: Configuration for LLM (api_key, model, temperature)
        """
        self.embedder = embedder
        self.index_builder = index_builder
        
        # Setup LLM
        llm_config = llm_config or {}
        api_key = llm_config.get('api_key', os.getenv('GOOGLE_API_KEY'))
        model_name = llm_config.get('model', os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'))
        temperature = llm_config.get('temperature', float(os.getenv('LLM_TEMPERATURE', '0.2')))
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or config")
        
        print(f"Initializing LLM: {model_name}")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )

        # self.llm = VegasChatLLM(
        #     prompt_id = "ANSIBLE_AGENT_PROMPT"
        # )
    
    def query(self, question: str, top_k: int = 5, min_similarity_threshold: float = 0.0) -> QueryResult:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            min_similarity_threshold: Minimum similarity score to include
        
        Returns:
            QueryResult with answer and sources
        """
        print(f"\n{'='*80}")
        print(f"Query: {question}")
        print(f"{'='*80}")
        
        # 1. Generate query embedding
        print("\n[1/3] Generating query embedding...")
        query_embedding = self.embedder.embed_query(question)
        
        # 2. Search for relevant chunks
        print(f"[2/3] Searching for top {top_k} relevant chunks...")
        distances, indices, metadata = self.index_builder.search(query_embedding, k=top_k)
        
        # Filter by similarity threshold (L2 distance)
        valid_results = []
        for dist, idx, meta in zip(distances, indices, metadata):
            if dist < float('inf'):  # Valid result
                valid_results.append((dist, idx, meta))
        
        if not valid_results:
            print("No relevant chunks found!")
            return QueryResult(
                answer="I couldn't find any relevant information in the repository to answer your question.",
                sources=[],
                query=question,
                context_used=""
            )
        
        print(f"Found {len(valid_results)} relevant chunks:")
        for i, (dist, idx, meta) in enumerate(valid_results[:5], 1):
            file_path = meta.get('metadata', {}).get('file_path', 'unknown')
            chunk_type = meta.get('chunk_type', 'unknown')
            print(f"  {i}. [{chunk_type}] {file_path} (distance: {dist:.4f})")
        
        # 3. Build context from retrieved chunks
        context = self._build_context(valid_results)
        
        # 4. Generate answer using LLM
        print("[3/3] Generating answer with LLM...")
        answer = self._generate_answer(question, context)
        
        # Prepare sources for response
        sources = []
        for dist, idx, meta in valid_results:
            source = {
                'file_path': meta.get('metadata', {}).get('file_path', 'unknown'),
                'chunk_type': meta.get('chunk_type', 'unknown'),
                'distance': float(dist),
                'content_preview': meta.get('content_preview', ''),
                'task_name': meta.get('metadata', {}).get('task_name'),
                'module_name': meta.get('metadata', {}).get('module_name')
            }
            sources.append(source)
        
        return QueryResult(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context[:500] + '...' if len(context) > 500 else context
        )
    
    def _build_context(self, results: List[Tuple[float, int, Dict]]) -> str:
        """
        Build context string from search results
        
        Args:
            results: List of (distance, index, metadata) tuples
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (dist, idx, meta) in enumerate(results, 1):
            file_path = meta.get('metadata', {}).get('file_path', 'unknown')
            chunk_type = meta.get('chunk_type', 'unknown')
            content_preview = meta.get('content_preview', '')
            
            # Build context entry
            context_entry = f"--- Source {i} [{chunk_type}] ---\n"
            context_entry += f"File: {file_path}\n"
            
            # Add task-specific info if available
            if chunk_type == 'task':
                task_name = meta.get('metadata', {}).get('task_name')
                module_name = meta.get('metadata', {}).get('module_name')
                if task_name:
                    context_entry += f"Task: {task_name}\n"
                if module_name:
                    context_entry += f"Module: {module_name}\n"
            
            context_entry += f"\nContent:\n{content_preview}\n"
            context_parts.append(context_entry)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            question: User question
            context: Retrieved context
        
        Returns:
            Generated answer
        """
        system_prompt = """You are an expert Ansible assistant. Your role is to answer questions about Ansible playbooks, roles, and configurations based on the provided context from a repository.

Guidelines:
1. Answer based ONLY on the provided context from the repository
2. If the context doesn't contain enough information, say so clearly
3. Cite specific files, tasks, or modules when relevant
4. Be concise but thorough
5. Use technical terminology appropriately
6. If you see task definitions, explain what they do
7. If you see variables or configurations, explain their purpose
"""
        
        user_prompt = f"""Based on the following context from an Ansible repository, please answer the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
    
    def print_result(self, result: QueryResult):
        """
        Pretty print a query result
        
        Args:
            result: QueryResult to print
        """
        print(f"\n{'='*80}")
        print("ANSWER")
        print(f"{'='*80}\n")
        print(result.answer)
        print(f"\n{'='*80}")
        print("SOURCES")
        print(f"{'='*80}\n")
        
        for i, source in enumerate(result.sources, 1):
            print(f"{i}. File: {source['file_path']}")
            print(f"   Type: {source['chunk_type']}")
            print(f"   Similarity: {1 / (1 + source['distance']):.3f}")
            if source.get('task_name'):
                print(f"   Task: {source['task_name']}")
            if source.get('module_name'):
                print(f"   Module: {source['module_name']}")
            print()


def main():
    """Test the query engine"""
    import argparse
    from embedder import Embedder
    from index_builder import IndexBuilder
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Query the RAG system')
    parser.add_argument('index_dir', help='Directory containing FAISS index')
    parser.add_argument('--query', '-q', help='Query to search for')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model')
    
    args = parser.parse_args()
    
    # Load embedder
    print("Loading embedding model...")
    embedder = Embedder(model_name=args.model)
    
    # Load index
    print("Loading FAISS index...")
    index_builder = IndexBuilder()
    index_builder.load_index(args.index_dir)
    
    # Create query engine
    query_engine = RAGQueryEngine(embedder, index_builder)
    
    if args.query:
        # Single query
        result = query_engine.query(args.query, top_k=args.top_k)
        query_engine.print_result(result)
    else:
        # Interactive mode
        print("\n=== RAG Query Engine ===")
        print("Type your questions (or 'quit' to exit)\n")
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = query_engine.query(question, top_k=args.top_k)
                query_engine.print_result(result)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()

