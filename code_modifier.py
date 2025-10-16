"""
Code Modifier Agent
Modifies code files with approval workflow
Shows planned changes and requires user approval before applying
"""

import os
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class FileModification:
    """Represents a planned modification to a file"""
    file_path: str
    original_content: str
    modified_content: str
    reason: str
    approved: bool = False
    applied: bool = False
    
    def get_diff(self) -> str:
        """Generate unified diff of changes"""
        original_lines = self.original_content.splitlines(keepends=True)
        modified_lines = self.modified_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}",
            lineterm=''
        )
        
        return ''.join(diff)
    
    def display(self):
        """Display the modification plan"""
        print(f"\n{'='*80}")
        print(f"FILE: {self.file_path}")
        print(f"{'='*80}")
        print(f"Reason: {self.reason}\n")
        print("Diff:")
        print(self.get_diff())
        print(f"{'='*80}\n")


class CodeModifierAgent:
    """Agent for modifying code with approval workflow"""
    
    def __init__(self, repo_path: str, rag_query_engine=None, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize code modifier agent
        
        Args:
            repo_path: Path to repository
            rag_query_engine: RAG query engine for context (optional)
            llm_config: LLM configuration
        """
        self.repo_path = Path(repo_path)
        self.rag_query_engine = rag_query_engine
        self.modifications: List[FileModification] = []
        
        # Setup LLM
        llm_config = llm_config or {}
        api_key = llm_config.get('api_key', os.getenv('GOOGLE_API_KEY'))
        model_name = llm_config.get('model', os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'))
        temperature = llm_config.get('temperature', float(os.getenv('LLM_TEMPERATURE', '0.2')))
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or config")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )
    
    def plan_modification(self, instruction: str, target_files: Optional[List[str]] = None) -> List[FileModification]:
        """
        Plan code modifications based on instruction
        
        Args:
            instruction: What to modify
            target_files: Specific files to modify (optional)
        
        Returns:
            List of planned modifications
        """
        print(f"\n{'='*80}")
        print("PLANNING CODE MODIFICATIONS")
        print(f"{'='*80}")
        print(f"Instruction: {instruction}\n")
        
        # Use RAG to find relevant files if not specified
        if not target_files and self.rag_query_engine:
            print("Searching for relevant files...")
            query_result = self.rag_query_engine.query(
                f"Which files are relevant for: {instruction}",
                top_k=5
            )
            target_files = list(set([
                source['file_path'] for source in query_result.sources
                if source['file_path'] != 'unknown'
            ]))
            print(f"Found {len(target_files)} relevant files")
        
        if not target_files:
            print("No target files specified or found")
            return []
        
        modifications = []
        
        for file_path in target_files:
            print(f"\nAnalyzing: {file_path}")
            
            full_path = self.repo_path / file_path
            if not full_path.exists():
                print(f"  ⚠️  File not found, skipping")
                continue
            
            # Read original content
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
                continue
            
            # Generate modification using LLM
            modification = self._generate_modification(
                file_path, original_content, instruction
            )
            
            if modification:
                modifications.append(modification)
                print(f"  ✅ Modification planned")
            else:
                print(f"  ℹ️  No changes needed")
        
        self.modifications.extend(modifications)
        return modifications
    
    def _generate_modification(self, file_path: str, original_content: str, 
                               instruction: str) -> Optional[FileModification]:
        """
        Generate modification for a file using LLM
        
        Args:
            file_path: Path to file
            original_content: Original file content
            instruction: Modification instruction
        
        Returns:
            FileModification or None
        """
        system_prompt = """You are an expert code modification assistant. Your task is to modify code files based on instructions.

Guidelines:
1. Analyze the original content carefully
2. Make ONLY the necessary changes to fulfill the instruction
3. Preserve all existing functionality unless explicitly asked to change
4. Maintain code style and formatting
5. If no changes are needed, respond with "NO_CHANGES_NEEDED"
6. Return ONLY the complete modified file content, nothing else

Format your response as:
REASON: <one line explanation of changes>
---
<complete modified file content>
"""
        
        user_prompt = f"""File: {file_path}

Instruction: {instruction}

Original Content:
{original_content}

Please provide the modified version of this file."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            if "NO_CHANGES_NEEDED" in response_text:
                return None
            
            # Parse response
            if "---" in response_text:
                parts = response_text.split("---", 1)
                reason_part = parts[0].strip()
                modified_content = parts[1].strip()
                
                # Extract reason
                reason = reason_part.replace("REASON:", "").strip()
                if not reason:
                    reason = "Modified based on instruction"
            else:
                modified_content = response_text.strip()
                reason = "Modified based on instruction"
            
            # Check if actually different
            if modified_content == original_content:
                return None
            
            return FileModification(
                file_path=file_path,
                original_content=original_content,
                modified_content=modified_content,
                reason=reason
            )
            
        except Exception as e:
            print(f"  ❌ Error generating modification: {e}")
            return None
    
    def review_modifications(self) -> None:
        """Display all planned modifications for review"""
        if not self.modifications:
            print("\nNo modifications planned")
            return
        
        print(f"\n{'='*80}")
        print(f"PLANNED MODIFICATIONS ({len(self.modifications)} files)")
        print(f"{'='*80}\n")
        
        for i, mod in enumerate(self.modifications, 1):
            print(f"\n[{i}/{len(self.modifications)}]")
            mod.display()
    
    def approve_all(self) -> None:
        """Approve all modifications"""
        for mod in self.modifications:
            mod.approved = True
        print(f"\n✅ Approved all {len(self.modifications)} modifications")
    
    def approve_modification(self, index: int) -> None:
        """Approve a specific modification by index"""
        if 0 <= index < len(self.modifications):
            self.modifications[index].approved = True
            print(f"✅ Approved modification {index + 1}")
        else:
            print(f"❌ Invalid index: {index}")
    
    def interactive_approval(self) -> None:
        """Interactive approval workflow"""
        if not self.modifications:
            print("\nNo modifications to approve")
            return
        
        print(f"\n{'='*80}")
        print("INTERACTIVE APPROVAL")
        print(f"{'='*80}\n")
        
        for i, mod in enumerate(self.modifications, 1):
            print(f"\n[{i}/{len(self.modifications)}]")
            mod.display()
            
            while True:
                response = input(f"Approve this modification? (y/n/q): ").strip().lower()
                
                if response == 'y':
                    mod.approved = True
                    print("✅ Approved")
                    break
                elif response == 'n':
                    print("❌ Rejected")
                    break
                elif response == 'q':
                    print("Stopping approval process")
                    return
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 'q' to quit")
        
        approved_count = sum(1 for m in self.modifications if m.approved)
        print(f"\n✅ Approved {approved_count}/{len(self.modifications)} modifications")
    
    def apply_modifications(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply approved modifications
        
        Args:
            dry_run: If True, don't actually write files
        
        Returns:
            Statistics about applied modifications
        """
        approved_mods = [m for m in self.modifications if m.approved and not m.applied]
        
        if not approved_mods:
            print("\nNo approved modifications to apply")
            return {'applied': 0, 'errors': 0}
        
        print(f"\n{'='*80}")
        print(f"APPLYING MODIFICATIONS {'(DRY RUN)' if dry_run else ''}")
        print(f"{'='*80}\n")
        
        stats = {'applied': 0, 'errors': 0, 'error_details': []}
        
        for mod in approved_mods:
            print(f"Applying: {mod.file_path}")
            
            if dry_run:
                print("  [DRY RUN] Would write file")
                stats['applied'] += 1
                continue
            
            try:
                full_path = self.repo_path / mod.file_path
                
                # Create backup
                backup_path = full_path.with_suffix(full_path.suffix + '.bak')
                with open(full_path, 'r', encoding='utf-8') as f:
                    with open(backup_path, 'w', encoding='utf-8') as bf:
                        bf.write(f.read())
                
                # Write modified content
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(mod.modified_content)
                
                mod.applied = True
                stats['applied'] += 1
                print(f"  ✅ Applied (backup: {backup_path.name})")
                
            except Exception as e:
                stats['errors'] += 1
                error_msg = f"Error applying {mod.file_path}: {e}"
                stats['error_details'].append(error_msg)
                print(f"  ❌ {error_msg}")
        
        print(f"\n{'='*80}")
        print(f"Applied: {stats['applied']} files")
        if stats['errors']:
            print(f"Errors: {stats['errors']}")
        print(f"{'='*80}\n")
        
        return stats
    
    def rollback_last(self) -> None:
        """Rollback the last applied modification"""
        applied_mods = [m for m in self.modifications if m.applied]
        
        if not applied_mods:
            print("No modifications to rollback")
            return
        
        last_mod = applied_mods[-1]
        full_path = self.repo_path / last_mod.file_path
        backup_path = full_path.with_suffix(full_path.suffix + '.bak')
        
        if backup_path.exists():
            with open(backup_path, 'r', encoding='utf-8') as f:
                with open(full_path, 'w', encoding='utf-8') as target:
                    target.write(f.read())
            
            last_mod.applied = False
            print(f"✅ Rolled back: {last_mod.file_path}")
        else:
            print(f"❌ Backup not found for: {last_mod.file_path}")


def main():
    """Test code modifier"""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Code modification agent')
    parser.add_argument('instruction', help='What to modify')
    parser.add_argument('--repo', default=os.getenv('REPO_PATH', './RHEL8-CIS'), help='Repository path')
    parser.add_argument('--files', nargs='+', help='Specific files to modify')
    parser.add_argument('--auto-approve', action='store_true', help='Auto-approve all changes')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (don\'t write files)')
    
    args = parser.parse_args()
    
    # Create modifier
    modifier = CodeModifierAgent(repo_path=args.repo)
    
    # Plan modifications
    modifications = modifier.plan_modification(args.instruction, target_files=args.files)
    
    if not modifications:
        print("\nNo modifications planned")
        return
    
    # Review
    modifier.review_modifications()
    
    # Approve
    if args.auto_approve:
        modifier.approve_all()
    else:
        modifier.interactive_approval()
    
    # Apply
    modifier.apply_modifications(dry_run=args.dry_run)
    
    print("\n✅ Code modification complete!")


if __name__ == '__main__':
    main()

