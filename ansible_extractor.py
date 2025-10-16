"""
Ansible Content Extractor
Extracts content from Ansible repositories using three chunking strategies:
1. File-level chunking: Embed entire files
2. Smart chunking: Split large files intelligently
3. Task-level chunking: Extract individual tasks using ansible-content-capture
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    """Represents a chunk of content to be embedded"""
    content: str
    metadata: Dict[str, Any]
    chunk_type: str  # 'file', 'smart', 'task'
    
    def to_dict(self):
        return {
            'content': self.content,
            'metadata': self.metadata,
            'chunk_type': self.chunk_type
        }


class AnsibleExtractor:
    """Extract content from Ansible repositories with multiple chunking strategies"""
    
    def __init__(self, repo_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.repo_path = Path(repo_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = {'.yml', '.yaml', '.j2', '.md', '.rst', '.txt', '.py'}
        
    def extract_all(self) -> List[Chunk]:
        """Extract content using all three strategies"""
        chunks = []
        
        print("Extracting content using file-level chunking...")
        chunks.extend(self.extract_file_level())
        
        print(f"Extracting content using smart chunking (size: {self.chunk_size}, overlap: {self.chunk_overlap})...")
        chunks.extend(self.extract_smart_chunks())
        
        print("Extracting content using task-level chunking...")
        chunks.extend(self.extract_task_level())
        
        print(f"Total chunks extracted: {len(chunks)}")
        return chunks
    
    def extract_file_level(self) -> List[Chunk]:
        """Extract entire files as single chunks"""
        chunks = []
        
        for file_path in self._walk_repository():
            try:
                content = self._read_file(file_path)
                if not content or len(content.strip()) == 0:
                    continue
                
                # Only use file-level for smaller files
                line_count = len(content.split('\n'))
                if line_count > 300:  # Skip large files for file-level chunking
                    continue
                
                chunk = Chunk(
                    content=content,
                    metadata={
                        'file_path': str(file_path.relative_to(self.repo_path)),
                        'file_type': file_path.suffix,
                        'line_count': line_count,
                        'strategy': 'file_level'
                    },
                    chunk_type='file'
                )
                chunks.append(chunk)
                
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
        
        print(f"  - File-level chunks: {len(chunks)}")
        return chunks
    
    def extract_smart_chunks(self) -> List[Chunk]:
        """Split large files into intelligent chunks with overlap"""
        chunks = []
        
        for file_path in self._walk_repository():
            try:
                content = self._read_file(file_path)
                if not content or len(content.strip()) == 0:
                    continue
                
                lines = content.split('\n')
                line_count = len(lines)
                
                # Only apply smart chunking to larger files
                if line_count <= 300:
                    continue
                
                # For YAML files, try to split by logical boundaries
                if file_path.suffix in {'.yml', '.yaml'}:
                    file_chunks = self._split_yaml_intelligently(content, file_path)
                else:
                    file_chunks = self._split_by_lines(content, file_path)
                
                chunks.extend(file_chunks)
                
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
        
        print(f"  - Smart chunks: {len(chunks)}")
        return chunks
    
    def extract_task_level(self) -> List[Chunk]:
        """Extract individual Ansible tasks and plays"""
        chunks = []
        
        for file_path in self._walk_repository():
            # Only process YAML files for task extraction
            if file_path.suffix not in {'.yml', '.yaml'}:
                continue
            
            try:
                content = self._read_file(file_path)
                if not content:
                    continue
                
                # Parse YAML
                yaml_data = yaml.safe_load(content)
                if not yaml_data:
                    continue
                
                # Extract tasks from various Ansible structures
                file_chunks = self._extract_tasks_from_yaml(yaml_data, file_path, content)
                chunks.extend(file_chunks)
                
            except yaml.YAMLError as e:
                # Skip files with YAML errors
                continue
            except Exception as e:
                print(f"Warning: Could not extract tasks from {file_path}: {e}")
                continue
        
        print(f"  - Task-level chunks: {len(chunks)}")
        return chunks
    
    def _walk_repository(self):
        """Walk through repository and yield supported files"""
        for root, dirs, files in os.walk(self.repo_path):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'venv', 'node_modules'}]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = Path(root) / file
                if file_path.suffix in self.supported_extensions:
                    yield file_path
    
    def _read_file(self, file_path: Path) -> str:
        """Read file content with encoding fallback"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with latin-1 encoding as fallback
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except:
                return ""
    
    def _split_by_lines(self, content: str, file_path: Path) -> List[Chunk]:
        """Split content by lines with overlap"""
        chunks = []
        lines = content.split('\n')
        
        start_line = 0
        chunk_id = 0
        
        while start_line < len(lines):
            end_line = min(start_line + self.chunk_size, len(lines))
            chunk_lines = lines[start_line:end_line]
            chunk_content = '\n'.join(chunk_lines)
            
            if chunk_content.strip():
                chunk = Chunk(
                    content=chunk_content,
                    metadata={
                        'file_path': str(file_path.relative_to(self.repo_path)),
                        'file_type': file_path.suffix,
                        'chunk_id': chunk_id,
                        'start_line': start_line + 1,
                        'end_line': end_line,
                        'strategy': 'smart_lines'
                    },
                    chunk_type='smart'
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next chunk with overlap
            start_line = end_line - self.chunk_overlap if end_line < len(lines) else end_line
            
            if start_line >= end_line:  # Prevent infinite loop
                break
        
        return chunks
    
    def _split_yaml_intelligently(self, content: str, file_path: Path) -> List[Chunk]:
        """Split YAML files by logical boundaries (tasks, plays, etc.)"""
        chunks = []
        lines = content.split('\n')
        
        # Find logical boundaries (items starting with '- name:', '- hosts:', etc.)
        boundaries = [0]
        for i, line in enumerate(lines):
            if line.strip().startswith('- name:') or line.strip().startswith('- hosts:'):
                boundaries.append(i)
        
        boundaries.append(len(lines))
        
        # If we found logical boundaries, split by them
        if len(boundaries) > 2:
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                
                chunk_lines = lines[start:end]
                chunk_content = '\n'.join(chunk_lines)
                
                if chunk_content.strip() and len(chunk_content) > 50:
                    chunk = Chunk(
                        content=chunk_content,
                        metadata={
                            'file_path': str(file_path.relative_to(self.repo_path)),
                            'file_type': file_path.suffix,
                            'chunk_id': i,
                            'start_line': start + 1,
                            'end_line': end,
                            'strategy': 'smart_yaml'
                        },
                        chunk_type='smart'
                    )
                    chunks.append(chunk)
        else:
            # Fall back to line-based chunking
            chunks = self._split_by_lines(content, file_path)
        
        return chunks
    
    def _extract_tasks_from_yaml(self, yaml_data: Any, file_path: Path, original_content: str) -> List[Chunk]:
        """Extract individual tasks from parsed YAML"""
        chunks = []
        
        # Handle different YAML structures
        if isinstance(yaml_data, list):
            # List of plays or tasks
            for idx, item in enumerate(yaml_data):
                if isinstance(item, dict):
                    chunk = self._create_task_chunk(item, file_path, idx, original_content)
                    if chunk:
                        chunks.append(chunk)
        
        elif isinstance(yaml_data, dict):
            # Single play or role structure
            # Check for tasks
            if 'tasks' in yaml_data and isinstance(yaml_data['tasks'], list):
                for idx, task in enumerate(yaml_data['tasks']):
                    chunk = self._create_task_chunk(task, file_path, idx, original_content, parent='tasks')
                    if chunk:
                        chunks.append(chunk)
            
            # Check for handlers
            if 'handlers' in yaml_data and isinstance(yaml_data['handlers'], list):
                for idx, handler in enumerate(yaml_data['handlers']):
                    chunk = self._create_task_chunk(handler, file_path, idx, original_content, parent='handlers')
                    if chunk:
                        chunks.append(chunk)
            
            # Check for pre_tasks, post_tasks
            for section in ['pre_tasks', 'post_tasks', 'block']:
                if section in yaml_data and isinstance(yaml_data[section], list):
                    for idx, task in enumerate(yaml_data[section]):
                        chunk = self._create_task_chunk(task, file_path, idx, original_content, parent=section)
                        if chunk:
                            chunks.append(chunk)
        
        return chunks
    
    def _create_task_chunk(self, task_data: Dict, file_path: Path, index: int, 
                          original_content: str, parent: str = 'root') -> Chunk:
        """Create a chunk from a task dictionary"""
        if not isinstance(task_data, dict):
            return None
        
        # Extract task information
        task_name = task_data.get('name', f'Unnamed task {index}')
        
        # Find the module being used
        module_name = None
        for key in task_data.keys():
            if key not in ['name', 'when', 'tags', 'with_items', 'loop', 'register', 
                          'become', 'become_user', 'vars', 'notify', 'changed_when',
                          'failed_when', 'ignore_errors', 'delegate_to']:
                module_name = key
                break
        
        # Create a readable representation of the task
        task_content = f"Task: {task_name}\n"
        if module_name:
            task_content += f"Module: {module_name}\n"
        
        task_content += f"File: {file_path.relative_to(self.repo_path)}\n"
        task_content += f"Section: {parent}\n\n"
        task_content += "Task Definition:\n"
        task_content += yaml.dump(task_data, default_flow_style=False, sort_keys=False)
        
        chunk = Chunk(
            content=task_content,
            metadata={
                'file_path': str(file_path.relative_to(self.repo_path)),
                'task_name': task_name,
                'module_name': module_name or 'unknown',
                'task_index': index,
                'parent_section': parent,
                'strategy': 'task_level'
            },
            chunk_type='task'
        )
        
        return chunk


def main():
    """Test the extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract content from Ansible repository')
    parser.add_argument('repo_path', help='Path to Ansible repository')
    parser.add_argument('--output', '-o', default='chunks.json', help='Output file for chunks')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for smart chunking')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap')
    
    args = parser.parse_args()
    
    extractor = AnsibleExtractor(args.repo_path, args.chunk_size, args.chunk_overlap)
    chunks = extractor.extract_all()
    
    # Save chunks to JSON
    chunks_data = [chunk.to_dict() for chunk in chunks]
    with open(args.output, 'w') as f:
        json.dump(chunks_data, f, indent=2)
    
    print(f"\nSaved {len(chunks)} chunks to {args.output}")
    
    # Print statistics
    print("\nChunk Statistics:")
    print(f"  File-level chunks: {sum(1 for c in chunks if c.chunk_type == 'file')}")
    print(f"  Smart chunks: {sum(1 for c in chunks if c.chunk_type == 'smart')}")
    print(f"  Task-level chunks: {sum(1 for c in chunks if c.chunk_type == 'task')}")


if __name__ == '__main__':
    main()

