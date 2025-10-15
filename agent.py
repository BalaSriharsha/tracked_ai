import os
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Annotated, TypedDict
from dotenv import load_dotenv
from git import Repo
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from pyvegas.langx.llm import VegasChatLLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Add ansible-content-capture to sys.path
ansible_capture_path = os.path.join(os.path.dirname(__file__), "ansible-content-capture", "src")
if os.path.exists(ansible_capture_path) and ansible_capture_path not in sys.path:
    sys.path.insert(0, ansible_capture_path)

# Configuration
REPO_LOCAL_PATH = "./RHEL8-CIS"
# GIT_REPO_URL = os.getenv("GIT_REPO_URL")

# Global storage for modification plans
PENDING_MODIFICATION_PLAN = None

# Global storage for last search results (file paths)
_LAST_SEARCH_PATHS = []

# Global thinking state
# The thinking tag system wraps all intermediate processing and reasoning in <thinking> tags
# Only the final answer is displayed outside the thinking tags
_THINKING_ACTIVE = False

# Configuration constants for display and preview limits
class Config:
    # Diff preview limits
    DIFF_PREVIEW_LINES = 10
    CONTENT_PREVIEW_CHARS = 500
    
    # Branch name limits
    BRANCH_NAME_MAX_LENGTH = 40
    
    # File reading limits
    LARGE_FILE_LINE_THRESHOLD = 500
    LARGE_FILE_PREVIEW_LINES = 100
    
    # File summary limits
    SUMMARY_PREVIEW_START_LINES = 10
    SUMMARY_PREVIEW_END_LINES = 5
    SUMMARY_TOTAL_LINES_THRESHOLD = 15
    
    # Search and analysis limits
    QUICK_SCAN_LINES = 200
    MAX_RELEVANT_FILES_DISPLAY = 15
    MAX_ITEMS_DISPLAY = 10
    MAX_GROUP_VARS_DISPLAY = 5
    
    # Verification limits
    VERIFICATION_PREVIEW_LINES = 30
    
    # Tool output limits
    TOOL_OUTPUT_PREVIEW_CHARS = 100
    MAX_PLAN_STEPS = 10
    
    # Display separator widths
    SEPARATOR_WIDTH_STANDARD = 80
    SEPARATOR_WIDTH_DIFF = 70

# # Clone or update repository
# def setup_repo():
#     """Clone or pull the Git repository."""
#     if os.path.exists(REPO_LOCAL_PATH):
#         print(f"Repository exists at {REPO_LOCAL_PATH}, pulling latest changes...")
#         repo = Repo(REPO_LOCAL_PATH)
#         repo.remotes.origin.pull()
#     else:
#         print(f"Cloning repository from {GIT_REPO_URL}...")
#         Repo.clone_from(GIT_REPO_URL, REPO_LOCAL_PATH)
#     print("Repository ready.")

# Helper functions for thinking tag management
def start_thinking() -> None:
    """Open the thinking tag for displaying all intermediate processing."""
    global _THINKING_ACTIVE
    if not _THINKING_ACTIVE:
        print("\n<thinking>", flush=True)
        _THINKING_ACTIVE = True

def end_thinking() -> None:
    """Close the thinking tag before displaying final answer."""
    global _THINKING_ACTIVE
    if _THINKING_ACTIVE:
        print("</thinking>\n", flush=True)
        _THINKING_ACTIVE = False

# Helper functions for displaying agent thinking
def print_thinking(message: str, prefix="THINKING") -> None:
    """Display agent's thinking process in real-time."""
    # Ensure we're inside thinking tag
    if not _THINKING_ACTIVE:
        start_thinking()
    print(f"[{prefix}] {message}", flush=True)

def print_section(title: str) -> None:
    """Print a section header."""
    # Ensure we're inside thinking tag
    if not _THINKING_ACTIVE:
        start_thinking()
    print(f"\n{'='*Config.SEPARATOR_WIDTH_STANDARD}")
    print(f"{title}")
    print(f"{'='*Config.SEPARATOR_WIDTH_STANDARD}\n", flush=True)

# Helper functions for interactive modification approval
def display_modification_plan(plan: dict) -> None:
    """Display modification plan in a formatted, user-friendly way."""
    # Ensure we're inside thinking tag
    if not _THINKING_ACTIVE:
        start_thinking()
    print("\n" + "="*Config.SEPARATOR_WIDTH_STANDARD)
    print("MODIFICATION PLAN")
    print("="*Config.SEPARATOR_WIDTH_STANDARD)
    print(f"\nDescription: {plan.get('modification_description', 'No description provided')}")
    
    files_to_modify = plan.get('files_to_modify', [])
    if files_to_modify:
        print(f"\nFiles to modify ({len(files_to_modify)}):")
        for idx, file_info in enumerate(files_to_modify, 1):
            file_path = file_info.get('file', 'Unknown file')
            changes = file_info.get('changes', 'No changes specified')
            print(f"\n  {idx}. {file_path}")
            print(f"     Changes: {changes}")
            
            # Show edit diff if available
            if file_info.get('edit_type') == 'edit' and file_info.get('old_string') and file_info.get('new_string'):
                old_str = file_info['old_string']
                new_str = file_info['new_string']
                print(f"\n     Diff:")
                print("     " + "-" * Config.SEPARATOR_WIDTH_DIFF)
                # Show old content with - prefix
                for line in old_str.split('\n')[:Config.DIFF_PREVIEW_LINES]:
                    print(f"     - {line}")
                if old_str.count('\n') > Config.DIFF_PREVIEW_LINES:
                    print(f"     ... ({old_str.count(chr(10)) - Config.DIFF_PREVIEW_LINES} more lines)")
                print("     " + "-" * Config.SEPARATOR_WIDTH_DIFF)
                # Show new content with + prefix
                for line in new_str.split('\n')[:Config.DIFF_PREVIEW_LINES]:
                    print(f"     + {line}")
                if new_str.count('\n') > Config.DIFF_PREVIEW_LINES:
                    print(f"     ... ({new_str.count(chr(10)) - Config.DIFF_PREVIEW_LINES} more lines)")
                print("     " + "-" * Config.SEPARATOR_WIDTH_DIFF)
    else:
        print("\nNo specific files listed in modification plan.")
    
    # Only show new_content if it's a full file replacement (deprecated approach)
    if plan.get('new_content') and not files_to_modify:
        print(f"\nWarning: Using full file replacement (not recommended)")
        print(f"\nNew Content Preview:")
        print("-" * Config.SEPARATOR_WIDTH_STANDARD)
        content_preview = plan['new_content'][:Config.CONTENT_PREVIEW_CHARS]
        print(content_preview)
        if len(plan['new_content']) > Config.CONTENT_PREVIEW_CHARS:
            print(f"\n... (truncated, {len(plan['new_content']) - Config.CONTENT_PREVIEW_CHARS} more characters)")
        print("-" * Config.SEPARATOR_WIDTH_STANDARD)
    
    print("\n" + "="*Config.SEPARATOR_WIDTH_STANDARD)

def request_user_approval() -> bool:
    """Request user approval for the modification plan."""
    while True:
        response = input("\nDo you approve these changes? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            print("Changes approved. Proceeding with modifications...\n")
            return True
        elif response in ['no', 'n']:
            print("Changes rejected. No modifications will be made.\n")
            return False
        else:
            print("Please enter 'yes' or 'no'.")

def detect_modification_type(description: str) -> str:
    """Detect the type of modification based on description keywords."""
    description_lower = description.lower()
    
    # Keywords for each type
    feature_keywords = ['add', 'implement', 'create', 'new', 'support', 'introduce']
    bugfix_keywords = ['fix', 'bug', 'issue', 'error', 'crash', 'problem', 'resolve']
    chore_keywords = ['update', 'upgrade', 'refactor', 'clean', 'maintenance', 'reorganize']
    hotfix_keywords = ['urgent', 'critical', 'security', 'hotfix', 'emergency']
    docs_keywords = ['document', 'readme', 'docs', 'comment', 'documentation']
    test_keywords = ['test', 'testing', 'spec', 'coverage']
    
    # Check for each type
    if any(keyword in description_lower for keyword in hotfix_keywords):
        return 'hotfix'
    elif any(keyword in description_lower for keyword in bugfix_keywords):
        return 'bugfix'
    elif any(keyword in description_lower for keyword in docs_keywords):
        return 'docs'
    elif any(keyword in description_lower for keyword in test_keywords):
        return 'test'
    elif any(keyword in description_lower for keyword in chore_keywords):
        return 'chore'
    elif any(keyword in description_lower for keyword in feature_keywords):
        return 'feature'
    else:
        return 'feature'  # Default to feature

def generate_branch_name(change_type: str, description: str) -> str:
    """Generate a branch name following Git Flow conventions."""
    import re
    
    # Convert description to kebab-case
    # Remove special characters and convert to lowercase
    clean_desc = re.sub(r'[^a-zA-Z0-9\s-]', '', description.lower())
    # Replace spaces with hyphens
    clean_desc = re.sub(r'\s+', '-', clean_desc.strip())
    # Remove multiple consecutive hyphens
    clean_desc = re.sub(r'-+', '-', clean_desc)
    # Limit length
    clean_desc = clean_desc[:Config.BRANCH_NAME_MAX_LENGTH].rstrip('-')
    
    # Create branch name
    branch_name = f"{change_type}/{clean_desc}"
    return branch_name

def ask_branch_creation(modification_description: str) -> tuple:
    """Ask user if they want to create a new branch for modifications.
    
    Returns:
        Tuple of (create_branch: bool, branch_name: str, change_type: str)
    """
    print("\n" + "="*Config.SEPARATOR_WIDTH_STANDARD)
    print("BRANCH CREATION")
    print("="*Config.SEPARATOR_WIDTH_STANDARD)
    
    # Detect modification type
    change_type = detect_modification_type(modification_description)
    print(f"\nDetected change type: {change_type}")
    
    # Generate proposed branch name
    proposed_name = generate_branch_name(change_type, modification_description)
    print(f"Proposed branch name: {proposed_name}")
    
    # Ask user
    while True:
        response = input("\nDo you want to create a new branch for these changes? (yes/no/custom): ").strip().lower()
        
        if response in ['no', 'n']:
            print("Proceeding with changes on current branch...\n")
            return (False, "", change_type)
        
        elif response in ['yes', 'y']:
            # Ask for confirmation of proposed name
            confirm = input(f"Use proposed branch name '{proposed_name}'? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                return (True, proposed_name, change_type)
            else:
                custom_name = input("Enter custom branch name: ").strip()
                if custom_name:
                    return (True, custom_name, change_type)
        
        elif response == 'custom':
            custom_name = input("Enter custom branch name: ").strip()
            if custom_name:
                return (True, custom_name, change_type)
        
        else:
            print("Please enter 'yes', 'no', or 'custom'.")

# Tool definitions
@tool
def list_files(directory: str = "") -> str:
    """List all files in the repository or a specific directory. Provide directory path relative to repo root."""
    base_path = Path(REPO_LOCAL_PATH) / directory
    if not base_path.exists():
        return f"Directory {directory} does not exist."
    
    files = []
    for item in base_path.rglob("*"):
        if item.is_file() and not str(item).startswith(f"{REPO_LOCAL_PATH}/.git"):
            rel_path = item.relative_to(REPO_LOCAL_PATH)
            files.append(str(rel_path))
    
    return "\n".join(files) if files else "No files found."

@tool
def read_file(file_path: str, start_line: int = None, end_line: int = None) -> str:
    """Read the content of a file from the repository.
    
    Args:
        file_path: Path to the file - can be absolute (starts with /) or relative to repo root
        start_line: Optional starting line number (1-indexed). If provided, only reads from this line
        end_line: Optional ending line number (1-indexed). If provided with start_line, reads only this range
    
    Returns:
        File content or the specified line range
    """
    # Handle both absolute and relative paths
    if file_path.startswith('/'):
        # Absolute path - use as-is
        full_path = Path(file_path)
    else:
        # Relative path - join with repo root
        full_path = Path(REPO_LOCAL_PATH) / file_path
    
    if not full_path.exists():
        return f"File {file_path} does not exist."
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            if start_line is not None:
                lines = f.readlines()
                total_lines = len(lines)
                
                # Adjust indices (convert to 0-indexed)
                start_idx = max(0, start_line - 1)
                end_idx = end_line if end_line else total_lines
                
                selected_lines = lines[start_idx:end_idx]
                content = ''.join(selected_lines)
                return f"Content of {file_path} (lines {start_line}-{end_idx}):\n{content}\n[Total file lines: {total_lines}]"
            else:
                content = f.read()
                line_count = content.count('\n') + 1
                
                # Warn if file is very large
                if line_count > Config.LARGE_FILE_LINE_THRESHOLD:
                    return f"Warning: {file_path} has {line_count} lines. Consider using start_line/end_line parameters to read specific sections.\n\nFirst {Config.LARGE_FILE_PREVIEW_LINES} lines:\n" + '\n'.join(content.split('\n')[:Config.LARGE_FILE_PREVIEW_LINES]) + f"\n\n... (truncated, {line_count - Config.LARGE_FILE_PREVIEW_LINES} more lines)"
                
                return f"Content of {file_path} ({line_count} lines):\n{content}"
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

@tool
def get_file_summary(file_path: str) -> str:
    """Get a quick summary of a file without reading full content. Shows file size, line count, and first few lines.
    Use this to decide if you need to read the full file.
    
    Args:
        file_path: Path to the file - can be absolute (starts with /) or relative to repo root
    
    Returns:
        File summary with metadata and preview
    """
    # Handle both absolute and relative paths
    if file_path.startswith('/'):
        full_path = Path(file_path)
    else:
        full_path = Path(REPO_LOCAL_PATH) / file_path
    
    if not full_path.exists():
        return f"File {file_path} does not exist."
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            total_lines = len(lines)
            
        # Get preview lines
        preview_start = ''.join(lines[:Config.SUMMARY_PREVIEW_START_LINES])
        preview_end = ''.join(lines[-Config.SUMMARY_PREVIEW_END_LINES:]) if total_lines > Config.SUMMARY_TOTAL_LINES_THRESHOLD else ""
        
        file_size = full_path.stat().st_size
        size_kb = file_size / 1024
        
        summary = f"""File: {file_path}
Size: {size_kb:.2f} KB
Lines: {total_lines}

--- First {Config.SUMMARY_PREVIEW_START_LINES} lines ---
{preview_start}"""
        
        if preview_end and total_lines > Config.SUMMARY_TOTAL_LINES_THRESHOLD:
            summary += f"\n... ({total_lines - Config.SUMMARY_TOTAL_LINES_THRESHOLD} lines omitted) ...\n\n--- Last {Config.SUMMARY_PREVIEW_END_LINES} lines ---\n{preview_end}"
        
        summary += f"\n\nUse read_file('{file_path}') to read full content, or read_file('{file_path}', start_line=X, end_line=Y) for specific lines."
        
        return summary
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

@tool
def find_relevant_files(query_description: str, file_pattern: str = "*.yml") -> str:
    """Identify files that are likely relevant to a query without reading their full content.
    This is a lightweight operation that helps narrow down which files to read.
    
    Args:
        query_description: Description of what you're looking for (e.g., 'nginx configuration', 'database setup')
        file_pattern: File glob pattern to search (e.g., '*.yml', 'roles/*/tasks/*.yml')
    
    Returns:
        List of potentially relevant files with brief context
    """
    base_path = Path(REPO_LOCAL_PATH)
    keywords = query_description.lower().split()
    relevant_files = []
    
    for file_path in base_path.rglob(file_pattern):
        if file_path.is_file() and ".git" not in str(file_path):
            rel_path = file_path.relative_to(REPO_LOCAL_PATH)
            path_str = str(rel_path).lower()
            
            # Check if path contains any keywords
            path_score = sum(1 for kw in keywords if kw in path_str)
            
            if path_score > 0:
                relevant_files.append((str(rel_path), path_score, "path match"))
                continue
            
            # Quick scan of file content for keywords
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read only first lines for efficiency
                    content_preview = ''.join(line for _, line in zip(range(Config.QUICK_SCAN_LINES), f))
                    content_lower = content_preview.lower()
                    
                    content_score = sum(1 for kw in keywords if kw in content_lower)
                    if content_score > 0:
                        relevant_files.append((str(rel_path), content_score, "content match"))
            except:
                pass
    
    # Sort by relevance score
    relevant_files.sort(key=lambda x: x[1], reverse=True)
    
    if not relevant_files:
        return f"No files found matching '{query_description}'. Try broader keywords or different file_pattern."
    
    output = [f"Found {len(relevant_files)} potentially relevant files for '{query_description}':\n"]
    for file, score, match_type in relevant_files[:Config.MAX_RELEVANT_FILES_DISPLAY]:
        output.append(f"  [{score} matches, {match_type}] {file}")
    
    if len(relevant_files) > Config.MAX_RELEVANT_FILES_DISPLAY:
        output.append(f"\n... and {len(relevant_files) - Config.MAX_RELEVANT_FILES_DISPLAY} more files")
    
    output.append(f"\nNext: Use get_file_summary() or grep_search() to explore these files, then read_file() for detailed content.")
    
    return '\n'.join(output)

@tool
def search_in_files(search_term: str, file_pattern: str = "*.yml") -> str:
    """Search for a term in files matching the pattern. Returns file paths and line numbers."""
    base_path = Path(REPO_LOCAL_PATH)
    results = []
    
    for file_path in base_path.rglob(file_pattern):
        if file_path.is_file() and ".git" not in str(file_path):
            try:
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if search_term.lower() in line.lower():
                            rel_path = file_path.relative_to(REPO_LOCAL_PATH)
                            results.append(f"{rel_path}:{line_num}: {line.strip()}")
            except:
                pass
    
    return "\n".join(results) if results else f"No matches found for '{search_term}'."

@tool
def grep_search(pattern: str, file_pattern: str = "*", case_sensitive: bool = False, max_results: int = 50) -> str:
    """Advanced grep-like search supporting regex patterns. Search for patterns across files.
    
    IMPORTANT: If this returns "No matches found", you MUST immediately call intelligent_search()
    with the same search term. intelligent_search will try naming variations automatically.
    
    Args:
        pattern: Regex pattern to search for (e.g., 'ansible.builtin.*', 'name:.*nginx', etc.)
        file_pattern: File glob pattern to search in (default: *, examples: '*.yml', '*.yaml', 'roles/**/tasks/*.yml')
        case_sensitive: Whether the search should be case-sensitive (default: False)
        max_results: Maximum number of results to return (default: 50)
    
    Returns:
        Matching lines with ABSOLUTE file paths and line numbers
    
    NOTE: If you get "No matches found", IMMEDIATELY try intelligent_search(pattern, file_pattern)
    """
    base_path = Path(REPO_LOCAL_PATH).resolve()
    results = []
    flags = 0 if case_sensitive else re.IGNORECASE
    
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Invalid regex pattern: {e}"
    
    # Search through files
    for file_path in base_path.rglob(file_pattern):
        if file_path.is_file() and ".git" not in str(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            # Return ABSOLUTE path to prevent hallucination
                            abs_path = str(file_path.resolve())
                            results.append(f"{abs_path}:{line_num}: {line.rstrip()}")
                            
                            if len(results) >= max_results:
                                results.append(f"\n... (truncated, showing first {max_results} results)")
                                return "\n".join(results)
            except Exception:
                pass
    
    if results:
        # Extract unique file paths and store globally
        global _LAST_SEARCH_PATHS
        file_paths = set()
        for result in results:
            if ':' in result and not result.startswith('...'):
                file_path = result.split(':')[0]
                file_paths.add(file_path)
        _LAST_SEARCH_PATHS = sorted(file_paths)
        
        return "\n".join(results) + f"\n\nFound {len(_LAST_SEARCH_PATHS)} unique files. Call read_all_found_files() to read all of them."
    else:
        return f"No matches found for pattern '{pattern}'."

@tool
def intelligent_search(search_term: str, file_pattern: str = "*.yml", max_results_per_pattern: int = 10) -> str:
    """Intelligent search that tries multiple naming pattern variations automatically.
    Use this when a simple search fails - it handles common naming variations like:
    - abc_def_ghi vs abcdef_ghi
    - snake_case vs camelCase
    - with/without prefixes or suffixes
    
    Args:
        search_term: The term to search for (e.g., 'abc_def_ghi')
        file_pattern: File glob pattern to search in (default: '*.yml')
        max_results_per_pattern: Maximum results to show per pattern variation (default: 10)
    
    Returns:
        Grouped results with ABSOLUTE file paths to prevent hallucination. Use these EXACT paths with read_file().
    """
    base_path = Path(REPO_LOCAL_PATH).resolve()
    
    # Generate naming pattern variations
    variations = []
    
    # Original
    variations.append(("Original", search_term))
    
    # Remove ALL underscores
    no_underscores = search_term.replace('_', '')
    if no_underscores != search_term:
        variations.append(("No underscores", no_underscores))
    
    # Try removing underscores between specific segments (common patterns)
    # E.g., abc_def_ -> abcdef_
    parts = search_term.split('_')
    if len(parts) >= 2:
        # Try combining first two parts
        combined_first_two = parts[0] + parts[1] + ('_' + '_'.join(parts[2:]) if len(parts) > 2 else '')
        if combined_first_two != search_term:
            variations.append(("Combined prefix", combined_first_two))
        
        # Try combining first three parts if available
        if len(parts) >= 3:
            combined_first_three = parts[0] + parts[1] + parts[2] + ('_' + '_'.join(parts[3:]) if len(parts) > 3 else '')
            if combined_first_three != search_term and combined_first_three != combined_first_two:
                variations.append(("Combined prefix (3 parts)", combined_first_three))
    
    # Try camelCase conversion
    camel_case = ''.join(word.capitalize() if i > 0 else word for i, word in enumerate(parts))
    if camel_case != search_term and camel_case != no_underscores:
        variations.append(("camelCase", camel_case))
    
    # Try with hyphens instead of underscores
    hyphenated = search_term.replace('_', '-')
    if hyphenated != search_term:
        variations.append(("Hyphenated", hyphenated))
    
    # Try as regex to catch partial matches (more flexible)
    # Convert underscores to optional pattern: abc_?def_?ghi
    flexible_pattern = search_term.replace('_', '_?')
    variations.append(("Flexible (regex)", flexible_pattern))
    
    # Store results grouped by variation
    results_by_variation = {}
    total_matches = 0
    
    for variation_name, pattern in variations:
        matches = []
        is_regex = variation_name == "Flexible (regex)"
        
        if is_regex:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue
        
        # Search through files
        for file_path in base_path.rglob(file_pattern):
            if file_path.is_file() and ".git" not in str(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            found = False
                            if is_regex:
                                found = regex.search(line) is not None
                            else:
                                found = pattern.lower() in line.lower()
                            
                            if found:
                                # Return ABSOLUTE path to prevent hallucination
                                abs_path = str(file_path.resolve())
                                matches.append(f"{abs_path}:{line_num}: {line.rstrip()}")
                                
                                if len(matches) >= max_results_per_pattern:
                                    break
                    
                    if len(matches) >= max_results_per_pattern:
                        break
                except Exception:
                    pass
        
        if matches:
            results_by_variation[variation_name] = {
                "pattern": pattern,
                "matches": matches,
                "count": len(matches)
            }
            total_matches += len(matches)
    
    # Format output
    if not results_by_variation:
        return f"No matches found for '{search_term}' or any of its variations.\n\nTried variations:\n" + "\n".join(f"  - {name}: {pattern}" for name, pattern in variations)
    
    output = [f"INTELLIGENT SEARCH RESULTS for '{search_term}'"]
    output.append(f"Found {total_matches} total matches across {len(results_by_variation)} pattern variations.\n")
    output.append("=" * Config.SEPARATOR_WIDTH_STANDARD)
    
    # Sort by number of matches (most relevant first)
    sorted_variations = sorted(results_by_variation.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for variation_name, data in sorted_variations:
        output.append(f"\n[{variation_name}] Pattern: '{data['pattern']}' - {data['count']} matches:")
        output.append("-" * Config.SEPARATOR_WIDTH_STANDARD)
        for match in data['matches']:
            output.append(f"  {match}")
        output.append("")
    
    output.append("=" * Config.SEPARATOR_WIDTH_STANDARD)
    
    # Extract unique file paths for clarity - MAKE THIS SUPER PROMINENT
    all_files = set()
    for data in results_by_variation.values():
        for match in data['matches']:
            file_path = match.split(':')[0]
            all_files.add(file_path)
    
    if all_files:
        output.append("\n" + "!" * Config.SEPARATOR_WIDTH_STANDARD)
        output.append("CRITICAL: FILE PATHS FOUND - USE THESE EXACT PATHS")
        output.append("DO NOT MODIFY OR ADD PREFIXES TO THESE PATHS")
        output.append("COPY AND PASTE EXACTLY AS SHOWN BELOW:")
        output.append("!" * Config.SEPARATOR_WIDTH_STANDARD)
        for fpath in sorted(all_files):
            output.append(f"\n>>> {fpath} <<<")
        output.append("\n" + "!" * Config.SEPARATOR_WIDTH_STANDARD)
        output.append("REMINDER: Use these paths EXACTLY as shown above with read_file()")
        output.append("Example: read_file('{}')".format(sorted(all_files)[0]))
        output.append("!" * Config.SEPARATOR_WIDTH_STANDARD)
    
    output.append("\nRECOMMENDATION: Review the matches above and select the most relevant pattern for your use case.")
    output.append("The pattern with the most matches is typically the correct one, but verify the context.")
    output.append("\n" + "=" * Config.SEPARATOR_WIDTH_STANDARD)
    output.append("NEXT STEP: Call read_all_found_files() to automatically read all files listed above.")
    output.append("This will show you the content of each file so you can decide which one to modify.")
    output.append("=" * Config.SEPARATOR_WIDTH_STANDARD)
    
    # Store paths for the read_all_found_files tool
    global _LAST_SEARCH_PATHS
    _LAST_SEARCH_PATHS = sorted(all_files)
    
    return "\n".join(output)

@tool
def read_all_found_files() -> str:
    """Read all files found by the last intelligent_search or grep_search call.
    This automatically reads every file that was found, so you can review their contents
    and decide which one to modify.
    
    IMPORTANT: Call this immediately after intelligent_search to see the content of all found files.
    
    Returns:
        Combined content of all files with clear separators between them.
    """
    global _LAST_SEARCH_PATHS
    
    if not _LAST_SEARCH_PATHS:
        return "No search results available. Call intelligent_search or grep_search first."
    
    output = []
    output.append("=" * Config.SEPARATOR_WIDTH_STANDARD)
    output.append(f"READING ALL {len(_LAST_SEARCH_PATHS)} FILES FOUND IN LAST SEARCH")
    output.append("=" * Config.SEPARATOR_WIDTH_STANDARD)
    
    for file_path in _LAST_SEARCH_PATHS:
        output.append("\n" + "=" * Config.SEPARATOR_WIDTH_STANDARD)
        output.append(f"FILE: {file_path}")
        output.append("=" * Config.SEPARATOR_WIDTH_STANDARD)
        
        # Read the file using absolute path
        if file_path.startswith('/'):
            full_path = Path(file_path)
        else:
            full_path = Path(REPO_LOCAL_PATH) / file_path
        
        if not full_path.exists():
            output.append(f"ERROR: File does not exist: {file_path}")
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                line_count = content.count('\n') + 1
                
                # Show full content for small files, preview for large files
                if line_count > Config.LARGE_FILE_LINE_THRESHOLD:
                    preview = '\n'.join(content.split('\n')[:Config.LARGE_FILE_PREVIEW_LINES])
                    output.append(f"\nContent ({line_count} lines, showing first {Config.LARGE_FILE_PREVIEW_LINES}):\n")
                    output.append(preview)
                    output.append(f"\n... ({line_count - Config.LARGE_FILE_PREVIEW_LINES} more lines)")
                else:
                    output.append(f"\nContent ({line_count} lines):\n")
                    output.append(content)
        except Exception as e:
            output.append(f"ERROR reading file: {str(e)}")
    
    output.append("\n" + "=" * Config.SEPARATOR_WIDTH_STANDARD)
    output.append("RECOMMENDATION: Review the files above and select which one to modify.")
    output.append("Then use create_modification_plan with the ABSOLUTE path of your chosen file.")
    output.append("=" * Config.SEPARATOR_WIDTH_STANDARD)
    
    return "\n".join(output)

@tool
def analyze_ansible_structure() -> str:
    """Analyze the Ansible repository structure and return information about roles, playbooks, tasks, handlers, and variables."""
    base_path = Path(REPO_LOCAL_PATH)
    
    if not base_path.exists():
        return "Repository not found. Please ensure the repository is cloned."
    
    analysis = {
        "playbooks": [],
        "roles": [],
        "inventory_files": [],
        "group_vars": [],
        "host_vars": [],
        "tasks_files": [],
        "handlers": [],
        "templates": [],
        "vars_files": []
    }
    
    # Find playbooks (YAML files in root or playbooks directory)
    for pattern in ["*.yml", "*.yaml", "playbooks/*.yml", "playbooks/*.yaml"]:
        for file_path in base_path.glob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(REPO_LOCAL_PATH)
                # Check if it's likely a playbook (contains 'hosts:' or 'import_playbook:')
                try:
                    with open(file_path, 'r') as f:
                        content = f.read(500)  # Read first 500 chars
                        if 'hosts:' in content or 'import_playbook:' in content:
                            analysis["playbooks"].append(str(rel_path))
                except:
                    pass
    
    # Find roles
    roles_path = base_path / "roles"
    if roles_path.exists():
        for role_dir in roles_path.iterdir():
            if role_dir.is_dir() and not role_dir.name.startswith('.'):
                role_info = {"name": role_dir.name, "components": []}
                
                # Check for standard role structure
                for component in ["tasks", "handlers", "templates", "files", "vars", "defaults", "meta"]:
                    component_path = role_dir / component
                    if component_path.exists():
                        role_info["components"].append(component)
                
                analysis["roles"].append(role_info)
    
    # Find inventory files
    for pattern in ["inventory/*", "hosts", "inventory.ini", "inventory.yml"]:
        for file_path in base_path.glob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(REPO_LOCAL_PATH)
                analysis["inventory_files"].append(str(rel_path))
    
    # Find group_vars and host_vars
    for var_type in ["group_vars", "host_vars"]:
        var_path = base_path / var_type
        if var_path.exists():
            for var_file in var_path.rglob("*.yml"):
                rel_path = var_file.relative_to(REPO_LOCAL_PATH)
                analysis[var_type].append(str(rel_path))
    
    # Find standalone tasks files
    for tasks_file in base_path.rglob("tasks/*.yml"):
        if tasks_file.is_file():
            rel_path = tasks_file.relative_to(REPO_LOCAL_PATH)
            analysis["tasks_files"].append(str(rel_path))
    
    # Format the output
    output = ["=== Ansible Repository Structure ===\n"]
    
    if analysis["playbooks"]:
        output.append(f"Playbooks ({len(analysis['playbooks'])}):")
        for pb in analysis["playbooks"][:Config.MAX_ITEMS_DISPLAY]:
            output.append(f"  - {pb}")
        if len(analysis["playbooks"]) > Config.MAX_ITEMS_DISPLAY:
            output.append(f"  ... and {len(analysis['playbooks']) - Config.MAX_ITEMS_DISPLAY} more")
        output.append("")
    
    if analysis["roles"]:
        output.append(f"Roles ({len(analysis['roles'])}):")
        for role in analysis["roles"][:Config.MAX_ITEMS_DISPLAY]:
            components = ", ".join(role["components"])
            output.append(f"  - {role['name']} ({components})")
        if len(analysis["roles"]) > Config.MAX_ITEMS_DISPLAY:
            output.append(f"  ... and {len(analysis['roles']) - Config.MAX_ITEMS_DISPLAY} more")
        output.append("")
    
    if analysis["inventory_files"]:
        output.append(f"Inventory Files ({len(analysis['inventory_files'])}):")
        for inv in analysis["inventory_files"]:
            output.append(f"  - {inv}")
        output.append("")
    
    if analysis["group_vars"]:
        output.append(f"Group Variables ({len(analysis['group_vars'])}):")
        for gv in analysis["group_vars"][:Config.MAX_GROUP_VARS_DISPLAY]:
            output.append(f"  - {gv}")
        if len(analysis["group_vars"]) > Config.MAX_GROUP_VARS_DISPLAY:
            output.append(f"  ... and {len(analysis['group_vars']) - Config.MAX_GROUP_VARS_DISPLAY} more")
        output.append("")
    
    return "\n".join(output) if any(analysis.values()) else "No Ansible structure detected in the repository."

@tool
def apply_file_edit(file_path: str, old_string: str, new_string: str, description: str = "") -> str:
    """Apply a surgical edit to a file by replacing old_string with new_string.
    This is the PREFERRED method for making file modifications as it only changes what's necessary.
    
    CRITICAL: file_path MUST be an ABSOLUTE path (starts with /)
    Get the path from read_all_found_files() output - NEVER construct paths manually!
    
    Args:
        file_path: ABSOLUTE path to the file (MUST start with / - get from read_all_found_files output)
        old_string: The exact string to find and replace (must match exactly including whitespace)
        new_string: The string to replace it with
        description: Description of what this edit accomplishes
    
    Returns:
        Status message about creating the modification plan
    
    Note:
        This creates a modification plan that will be shown to the user for approval.
        The edit will only be applied after user approval via execute_modification_plan.
    """
    global PENDING_MODIFICATION_PLAN, _LAST_SEARCH_PATHS
    
    # CRITICAL PATH VALIDATION - Prevents hallucination
    if not file_path.startswith('/'):
        return f"""
ERROR: INVALID FILE PATH - Path must be ABSOLUTE (start with /)

You provided: {file_path}
This is a RELATIVE or HALLUCINATED path!

REQUIRED ACTION:
1. Call read_all_found_files() to see all files from your last search
2. Copy the ABSOLUTE path from the file listing (starts with /)
3. Call apply_file_edit again with the ABSOLUTE path

Example of correct absolute path:
  /Users/balasriharshach/BalaSriharsha/github_agent/RHEL8-CIS/vars/main.yml
"""
    
    # Check if file exists
    file_obj = Path(file_path)
    if not file_obj.exists():
        available_paths = "\n".join(f"  - {p}" for p in _LAST_SEARCH_PATHS) if _LAST_SEARCH_PATHS else "  (No paths available)"
        return f"""
ERROR: FILE DOES NOT EXIST!

You provided: {file_path}
This file does NOT exist!

Available paths from last search:
{available_paths}

REQUIRED ACTION:
1. Call read_all_found_files() to see all files
2. Copy the EXACT absolute path shown in the output
3. Call apply_file_edit again with that path
"""
    
    # Check if path is from discovered paths (warning if not)
    if _LAST_SEARCH_PATHS and file_path not in _LAST_SEARCH_PATHS:
        available_paths = "\n".join(f"  - {p}" for p in _LAST_SEARCH_PATHS)
        return f"""
WARNING: File path was not in the list of discovered paths!

You provided: {file_path}

This path was NOT found by the most recent search. It may be a hallucinated or incorrect path.

Paths discovered by last search:
{available_paths}

REQUIRED ACTION:
1. Verify that you copied the path EXACTLY from read_all_found_files() output
2. If this path was NOT in that output, you may have hallucinated it
3. Go back and use one of the paths listed above

ONLY use paths that were discovered by intelligent_search or grep_search!
"""
    
    # Validate old_string exists in file (prevents content hallucination)
    if old_string:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                actual_content = f.read()
            
            if old_string not in actual_content:
                # Show the actual file content (no truncation for small files)
                if len(actual_content) < 2000:
                    content_display = actual_content
                else:
                    content_display = actual_content[:2000] + f"\n\n... ({len(actual_content) - 2000} more characters)"
                
                return f"""
ERROR: CONTENT HALLUCINATION DETECTED!

The old_string you provided does NOT exist in the file!

File: {file_path}

Your old_string:
{old_string}

ACTUAL file content:
{content_display}

IMMEDIATE RETRY REQUIRED:
You MUST retry apply_file_edit NOW in THIS SAME STEP with the correct content!

1. Look at the ACTUAL file content shown above
2. Find the EXACT lines you want to modify
3. Copy those EXACT lines (with correct spacing, quotes, indentation) as old_string
4. Create new_string with your modification
5. Call apply_file_edit AGAIN with the correct old_string and new_string

DO NOT move to the next step!
DO NOT guess or generate content!
RETRY NOW with EXACT content from above!
"""
        except Exception as e:
            return f"ERROR: Could not read file {file_path}: {str(e)}"
    
    files_to_modify = [{
        "file": file_path,
        "changes": description,
        "edit_type": "edit",
        "old_string": old_string,
        "new_string": new_string
    }]
    
    plan = {
        "modification_description": description,
        "status": "pending_approval",
        "files_to_modify": files_to_modify,
        "instructions": "Call execute_modification_plan to show this plan to the user and request approval."
    }
    
    PENDING_MODIFICATION_PLAN = plan
    return f"Edit plan created for {file_path}. Call execute_modification_plan to show the plan to the user, request approval, and apply the changes."

@tool
def create_modification_plan(description: str, file_path: str = "", changes_summary: str = "", old_string: str = "", new_string: str = "") -> str:
    """Create a plan for surgical code modifications. This prepares the plan but does NOT request approval yet.
    Approval will be requested when execute_modification_plan is called.
    
    CRITICAL: file_path MUST be an ABSOLUTE path (starts with /)
    Get the path from read_all_found_files() output - NEVER construct paths manually!
    
    IMPORTANT: This tool now supports SURGICAL EDITS. Always prefer using old_string/new_string over full rewrites.
    
    Args:
        description: Description of what needs to be changed
        file_path: ABSOLUTE path to the file (MUST start with / - get from read_all_found_files output)
        changes_summary: Summary of changes to be made (optional)
        old_string: The exact string to find and replace in the file (for surgical edits) (optional)
        new_string: The string to replace old_string with (for surgical edits) (optional)
    
    Returns:
        Status message confirming the plan was created
    
    Note:
        ALWAYS use old_string and new_string for targeted edits instead of rewriting entire files.
        This tool only PREPARES the plan. Call execute_modification_plan to show the plan, request approval, and execute.
    """
    global PENDING_MODIFICATION_PLAN, _LAST_SEARCH_PATHS
    
    # CRITICAL PATH VALIDATION - Prevents hallucination
    if file_path:
        # Check 1: Must be absolute path
        if not file_path.startswith('/'):
            return f"""
ERROR: INVALID FILE PATH - Path must be ABSOLUTE (start with /)

You provided: {file_path}
This is a RELATIVE or HALLUCINATED path!

REQUIRED ACTION:
1. Call intelligent_search() to find the file
2. Call read_all_found_files() to see all found files
3. Copy the ABSOLUTE path from the file listing (starts with /)
4. Call create_modification_plan again with the ABSOLUTE path

Example of correct absolute path:
  /Users/balasriharshach/BalaSriharsha/github_agent/RHEL8-CIS/vars/main.yml

NEVER use paths like:
  - roles/something/vars/main.yml
  - vars/main.yml
  - /usr/local/google/home/...  (this is not your system!)
"""
        
        # Check 2: File must exist
        file_obj = Path(file_path)
        if not file_obj.exists():
            # Show available paths from last search
            available_paths = "\n".join(f"  - {p}" for p in _LAST_SEARCH_PATHS) if _LAST_SEARCH_PATHS else "  (No paths available - run intelligent_search first)"
            
            return f"""
ERROR: FILE DOES NOT EXIST - Path is hallucinated or incorrect!

You provided: {file_path}
This file does NOT exist on the system!

Available paths from last search:
{available_paths}

REQUIRED ACTION:
1. Call read_all_found_files() to see all files from your last search
2. Review the file contents shown in the output
3. Select the correct file based on its content
4. Copy the EXACT absolute path shown in the output
5. Call create_modification_plan again with that path

DO NOT construct paths manually!
DO NOT guess directory structures!
ONLY use paths from read_all_found_files() output!
"""
        
        # Check 3: Path should be from discovered paths (warning if not)
        if _LAST_SEARCH_PATHS and file_path not in _LAST_SEARCH_PATHS:
            # Check if this path is a valid file but just wasn't in the search results
            # This could happen if the user found the file through other means
            available_paths = "\n".join(f"  - {p}" for p in _LAST_SEARCH_PATHS)
            return f"""
WARNING: File path was not in the list of discovered paths!

You provided: {file_path}

This path was NOT found by the most recent search. It may be a hallucinated or incorrect path.

Paths discovered by last search:
{available_paths}

REQUIRED ACTION:
1. Verify that you copied the path EXACTLY from read_all_found_files() output
2. If this path was NOT in that output, you may have hallucinated it
3. Go back and use one of the paths listed above
4. If you need a different file, run a new search first

ONLY use paths that were discovered by intelligent_search or grep_search!
"""
    
    files_to_modify = []
    if file_path:
        # Check 4: Validate old_string exists in file (prevents content hallucination)
        if old_string:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    actual_content = f.read()
                
                if old_string not in actual_content:
                    # Show the actual file content (no truncation for small files)
                    if len(actual_content) < 2000:
                        content_display = actual_content
                    else:
                        content_display = actual_content[:2000] + f"\n\n... ({len(actual_content) - 2000} more characters)"
                    
                    return f"""
ERROR: CONTENT HALLUCINATION DETECTED!

The old_string you provided does NOT exist in the file!

File: {file_path}

Your old_string:
{old_string}

ACTUAL file content:
{content_display}

IMMEDIATE RETRY REQUIRED:
You MUST retry create_modification_plan NOW in THIS SAME STEP with the correct content!

1. Look at the ACTUAL file content shown above
2. Find the EXACT lines you want to modify
3. Copy those EXACT lines (with correct spacing, quotes, indentation) as old_string
4. Create new_string with your modification
5. Call create_modification_plan AGAIN with the correct old_string and new_string

DO NOT move to the next step!
DO NOT guess or generate content!
RETRY NOW with EXACT content from above!
"""
            except Exception as e:
                return f"ERROR: Could not read file {file_path}: {str(e)}"
        
        file_mod = {
            "file": file_path,
            "changes": changes_summary
        }
        
        # Surgical edit approach (PREFERRED)
        if old_string and new_string:
            file_mod["edit_type"] = "edit"
            file_mod["old_string"] = old_string
            file_mod["new_string"] = new_string
        # Deletion approach
        elif old_string and not new_string:
            file_mod["edit_type"] = "delete_content"
            file_mod["old_string"] = old_string
            file_mod["new_string"] = ""
        
        files_to_modify.append(file_mod)
    
    plan = {
        "modification_description": description,
        "status": "pending_approval",
        "files_to_modify": files_to_modify,
        "instructions": "Call execute_modification_plan to show this plan to the user and request approval."
    }
    
    # Store the plan (will be shown to user when execute_modification_plan is called)
    PENDING_MODIFICATION_PLAN = plan
    
    return f"Modification plan created successfully. Call execute_modification_plan to show the plan to the user, request approval, and apply the changes."

@tool
def verify_modification(file_path: str, expected_content_description: str, search_pattern: str = "") -> str:
    """Verify that a modification was applied correctly by checking the file content.
    
    MANDATORY: You MUST call this tool after execute_modification_plan completes successfully.
    This tool helps you verify if your changes were applied as expected.
    
    Args:
        file_path: Path to the modified file - can be absolute or relative to repo root
        expected_content_description: What you expect to find in the file after modification
        search_pattern: Optional specific pattern/text to search for in the modified file
    
    Returns:
        Verification result indicating whether the change was applied correctly.
        If verification fails, you should create a new modification plan and try again.
    
    Example:
        After adding a value to a list, call:
        verify_modification('/absolute/path/to/file.yml', 'value should be in the list', 'value')
    """
    # Handle both absolute and relative paths
    if file_path.startswith('/'):
        full_path = Path(file_path)
    else:
        full_path = Path(REPO_LOCAL_PATH) / file_path
    
    if not full_path.exists():
        return f"VERIFICATION FAILED: File {file_path} does not exist!"
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # If search pattern provided, check if it exists
        if search_pattern:
            if search_pattern in content:
                return f"VERIFICATION SUCCESSFUL: Found '{search_pattern}' in {file_path}.\n\nExpected: {expected_content_description}\nResult: Pattern found in file.\n\nThe modification was applied correctly!"
            else:
                # Pattern not found - modification failed
                return f"VERIFICATION FAILED: Pattern '{search_pattern}' NOT found in {file_path}!\n\nExpected: {expected_content_description}\nResult: Pattern missing from file.\n\nACTION REQUIRED: The modification did not work as expected. You MUST:\n1. Read the file to see its current state\n2. Create a NEW modification plan with corrected parameters\n3. Execute the new plan\n4. Verify again\n\nDo NOT give up - retry until the modification is correct!"
        else:
            # No specific pattern - just show file content for manual verification
            line_count = content.count('\n') + 1
            preview_lines = Config.VERIFICATION_PREVIEW_LINES
            preview = '\n'.join(content.split('\n')[:preview_lines])
            
            return f"FILE CONTENT PREVIEW for {file_path} ({line_count} lines):\n{'-'*80}\n{preview}\n{'-'*80}\n\nExpected: {expected_content_description}\n\nPlease review the content above and determine if the modification was successful.\nIf not, create a new modification plan and try again."
            
    except Exception as e:
        return f"VERIFICATION ERROR: Could not read {file_path}: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """DEPRECATED: Use create_modification_plan + execute_modification_plan instead.
    
    This tool bypasses the approval workflow and should NOT be used.
    For ALL file modifications, use:
    1. create_modification_plan() to prepare the change
    2. execute_modification_plan() to show plan, get approval, and execute
    
    This ensures consistent user approval for all modifications.
    """
    return "ERROR: write_file is deprecated. Use create_modification_plan followed by execute_modification_plan to ensure user approval for all modifications."

@tool
def execute_modification_plan() -> str:
    """Show the modification plan to the user, request approval, and execute if approved.
    
    This tool handles the complete approval workflow:
    1. Displays the modification plan
    2. Asks about branch creation
    3. Requests user approval
    4. Creates branch if requested
    5. Executes the changes (file modifications or deletions)
    
    IMPORTANT: After this tool succeeds, you MUST call verify_modification() to confirm
    the changes were applied correctly. If verification fails, retry with a new plan.
    
    Returns:
        Status message with results of all operations performed
    """
    global PENDING_MODIFICATION_PLAN
    
    if PENDING_MODIFICATION_PLAN is None:
        return "No modification plan found. Create one first using create_modification_plan."
    
    plan = PENDING_MODIFICATION_PLAN
    
    # Display the plan in the terminal
    display_modification_plan(plan)
    
    # Ask about branch creation
    description = plan.get("modification_description", "")
    create_branch, branch_name, change_type = ask_branch_creation(description)
    
    # Store branch info in plan
    plan["create_branch"] = create_branch
    plan["branch_name"] = branch_name
    plan["change_type"] = change_type
    
    # Request modification approval
    approved = request_user_approval()
    
    if not approved:
        # User rejected the plan
        plan["status"] = "rejected"
        PENDING_MODIFICATION_PLAN = None
        return "Modification plan rejected by user. No changes will be made."
    
    # User approved - create branch if requested
    if create_branch:
        try:
            repo = Repo(REPO_LOCAL_PATH)
            current_branch = repo.active_branch.name
            
            # Check if branch already exists
            if branch_name not in [b.name for b in repo.heads]:
                new_branch = repo.create_head(branch_name)
                new_branch.checkout()
                print(f"Created and switched to branch '{branch_name}' from '{current_branch}'\n")
            else:
                print(f"Branch '{branch_name}' already exists. Staying on current branch.\n")
        except Exception as e:
            print(f"Warning: Could not create branch: {str(e)}\n")
    
    # Mark as approved and execute
    plan["status"] = "approved"
    
    results = []
    
    # Handle files_to_modify
    for file_info in plan.get("files_to_modify", []):
        file_path = file_info.get("file")
        changes = file_info.get("changes")
        edit_type = file_info.get("edit_type", "full_replace")
        
        # Handle surgical edits (PREFERRED)
        if edit_type == "edit" or edit_type == "delete_content":
            old_string = file_info.get("old_string")
            new_string = file_info.get("new_string")
            
            if not old_string:
                results.append(f"Error: {file_path} - old_string is required for edit")
                continue
            
            try:
                # Handle both absolute and relative paths
                if file_path.startswith('/'):
                    full_path = Path(file_path)
                else:
                    full_path = Path(REPO_LOCAL_PATH) / file_path
                
                if not full_path.exists():
                    results.append(f"Error: {file_path} does not exist")
                    continue
                
                # Read current content
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if old_string exists in file
                if old_string not in content:
                    results.append(f"Error: {file_path} - old_string not found in file. The file may have changed.")
                    continue
                
                # Count occurrences
                occurrences = content.count(old_string)
                if occurrences > 1:
                    results.append(f"Error: {file_path} - old_string appears {occurrences} times in file. It must be unique.")
                    continue
                
                # Apply the edit
                new_content = content.replace(old_string, new_string, 1)
                
                # Write back
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                results.append(f"Successfully edited {file_path}: {changes}")
            except Exception as e:
                results.append(f"Error editing {file_path}: {str(e)}")
        
        # Handle full file replacement (DEPRECATED - for backward compatibility)
        elif edit_type == "full_replace":
            content = file_info.get("content")
            
            # Check if this is a deletion request
            is_deletion = content == "" and changes and any(keyword in changes.lower() for keyword in ['delete', 'deletion', 'remove', 'removal'])
            
            if is_deletion:
                # Delete the file
                try:
                    # Handle both absolute and relative paths
                    if file_path.startswith('/'):
                        full_path = Path(file_path)
                    else:
                        full_path = Path(REPO_LOCAL_PATH) / file_path
                    
                    if full_path.exists():
                        full_path.unlink()
                        results.append(f"Successfully deleted {file_path}")
                    else:
                        results.append(f"File not found (already deleted): {file_path}")
                except Exception as e:
                    results.append(f"Error deleting {file_path}: {str(e)}")
            elif content:
                # Write the new content to the file
                try:
                    # Handle both absolute and relative paths
                    if file_path.startswith('/'):
                        full_path = Path(file_path)
                    else:
                        full_path = Path(REPO_LOCAL_PATH) / file_path
                    
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(full_path, 'w') as f:
                        f.write(content)
                    results.append(f"Modified {file_path}: {changes}")
                except Exception as e:
                    results.append(f"Error modifying {file_path}: {str(e)}")
            else:
                results.append(f"{file_path}: {changes} (no content provided, skipped)")
    
    # Clear the pending plan after execution
    PENDING_MODIFICATION_PLAN = None
    
    if results:
        return "Modification Results:\n" + "\n".join(results)
    else:
        return "No modifications executed. The plan may not have contained executable changes."

# Git workflow tools
@tool
def git_fetch_all() -> str:
    """Fetch latest changes from all remotes to sync with remote repository."""
    try:
        repo = Repo(REPO_LOCAL_PATH)
        print_thinking("Fetching latest changes from all remotes...", "GIT")
        repo.remotes.origin.fetch()
        return "Successfully fetched latest changes from remote repository."
    except Exception as e:
        return f"Error fetching from remote: {str(e)}"

@tool
def git_get_current_branch() -> str:
    """Get the name of the current active branch."""
    try:
        repo = Repo(REPO_LOCAL_PATH)
        current_branch = repo.active_branch.name
        return f"Current branch: {current_branch}"
    except Exception as e:
        return f"Error getting current branch: {str(e)}"

@tool
def git_get_base_branch() -> str:
    """Determine the base/parent branch of the current branch."""
    try:
        repo = Repo(REPO_LOCAL_PATH)
        current_branch = repo.active_branch
        
        # Try to get tracking branch
        if current_branch.tracking_branch():
            tracking = current_branch.tracking_branch().name
            base = tracking.split('/')[-1] if '/' in tracking else tracking
            return f"Base branch: {base}"
        
        # Fallback: check if main or master exists
        branch_names = [b.name for b in repo.heads]
        if 'main' in branch_names:
            return "Base branch: main"
        elif 'master' in branch_names:
            return "Base branch: master"
        else:
            return "Base branch: Unable to determine (defaulting to main)"
    except Exception as e:
        return f"Error determining base branch: {str(e)}"

@tool
def git_sync_with_base(base_branch: str = "main") -> str:
    """Merge the base branch into current branch to sync with latest changes."""
    try:
        repo = Repo(REPO_LOCAL_PATH)
        current_branch = repo.active_branch.name
        
        print_thinking(f"Syncing {current_branch} with {base_branch}...", "GIT")
        
        # Check if base branch exists
        if base_branch not in [b.name for b in repo.heads]:
            return f"Base branch '{base_branch}' does not exist."
        
        # Merge base branch into current branch
        repo.git.merge(base_branch, '--no-edit')
        return f"Successfully merged {base_branch} into {current_branch}"
    except Exception as e:
        if "CONFLICT" in str(e):
            return f"Merge conflict detected. Please resolve conflicts manually: {str(e)}"
        return f"Error syncing with base branch: {str(e)}"

@tool
def git_create_branch(branch_name: str, from_current: bool = True) -> str:
    """Create a new branch and switch to it.
    
    Args:
        branch_name: Name of the new branch to create
        from_current: If True, create from current branch; if False, from base branch
    
    Returns:
        Status message about branch creation
    """
    try:
        repo = Repo(REPO_LOCAL_PATH)
        current_branch = repo.active_branch.name
        
        # Check if branch already exists
        if branch_name in [b.name for b in repo.heads]:
            return f"Branch '{branch_name}' already exists. Switch to it using git checkout."
        
        print_thinking(f"Creating new branch '{branch_name}' from '{current_branch}'...", "GIT")
        
        # Create and checkout new branch
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
        
        return f"Successfully created and switched to branch '{branch_name}' from '{current_branch}'"
    except Exception as e:
        return f"Error creating branch: {str(e)}"

@tool
def git_get_status() -> str:
    """Get current Git repository status including modified and staged files."""
    try:
        repo = Repo(REPO_LOCAL_PATH)
        status_lines = []
        
        # Get current branch
        status_lines.append(f"On branch: {repo.active_branch.name}")
        
        # Get modified files
        modified = [item.a_path for item in repo.index.diff(None)]
        if modified:
            status_lines.append(f"\nModified files ({len(modified)}):")
            for file in modified[:Config.MAX_ITEMS_DISPLAY]:
                status_lines.append(f"  - {file}")
            if len(modified) > Config.MAX_ITEMS_DISPLAY:
                status_lines.append(f"  ... and {len(modified) - Config.MAX_ITEMS_DISPLAY} more")
        
        # Get staged files
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        if staged:
            status_lines.append(f"\nStaged files ({len(staged)}):")
            for file in staged[:Config.MAX_ITEMS_DISPLAY]:
                status_lines.append(f"  - {file}")
            if len(staged) > Config.MAX_ITEMS_DISPLAY:
                status_lines.append(f"  ... and {len(staged) - Config.MAX_ITEMS_DISPLAY} more")
        
        # Get untracked files
        untracked = repo.untracked_files
        if untracked:
            status_lines.append(f"\nUntracked files ({len(untracked)}):")
            for file in untracked[:Config.MAX_ITEMS_DISPLAY]:
                status_lines.append(f"  - {file}")
            if len(untracked) > Config.MAX_ITEMS_DISPLAY:
                status_lines.append(f"  ... and {len(untracked) - Config.MAX_ITEMS_DISPLAY} more")
        
        if not modified and not staged and not untracked:
            status_lines.append("\nWorking tree clean")
        
        return "\n".join(status_lines)
    except Exception as e:
        return f"Error getting git status: {str(e)}"

@tool
def git_list_branches() -> str:
    """List all local branches with current branch marker."""
    try:
        repo = Repo(REPO_LOCAL_PATH)
        current_branch = repo.active_branch.name
        branches = []
        
        for branch in repo.heads:
            marker = "* " if branch.name == current_branch else "  "
            branches.append(f"{marker}{branch.name}")
        
        return "Local branches:\n" + "\n".join(branches)
    except Exception as e:
        return f"Error listing branches: {str(e)}"

# Ansible Content Capture tools
@tool
def scan_ansible_content(target_path: str = "") -> str:
    """Scan and analyze Ansible content (playbooks, roles, tasks) to extract detailed information.
    
    Args:
        target_path: Path to Ansible content relative to repo root (playbook, role dir, or project dir)
    
    Returns:
        JSON string with scanned content details including tasks, modules, variables, and structure
    """
    try:
        from ansible_content_capture.scanner import AnsibleScanner
        
        # Determine full path
        if target_path:
            full_path = os.path.join(REPO_LOCAL_PATH, target_path)
        else:
            full_path = REPO_LOCAL_PATH
        
        if not os.path.exists(full_path):
            return f"Error: Path does not exist: {target_path}"
        
        # Create scanner and run
        scanner = AnsibleScanner()
        scanner.silent = True
        result = scanner.run(target_dir=full_path)
        
        # Extract useful information
        scan_data = scanner.scan_records
        
        # Format output
        output = {
            "scanned_path": target_path or REPO_LOCAL_PATH,
            "projects": list(scan_data.get("project_file_list", {}).keys()),
            "total_files_scanned": len(scan_data.get("file_inventory", {})),
            "summary": "Scan completed successfully"
        }
        
        return json.dumps(output, indent=2)
    except ImportError:
        return "Error: ansible-content-capture is not properly installed. Please ensure the repository is cloned."
    except Exception as e:
        return f"Error scanning Ansible content: {str(e)}"

@tool
def extract_playbook_tasks(playbook_path: str) -> str:
    """Extract tasks and execution flow from an Ansible playbook.
    
    Args:
        playbook_path: Path to the playbook file relative to repo root
    
    Returns:
        JSON string with tasks, roles, and execution flow information
    """
    try:
        from ansible_content_capture.scanner import AnsibleScanner
        
        full_path = os.path.join(REPO_LOCAL_PATH, playbook_path)
        
        if not os.path.exists(full_path):
            return f"Error: Playbook not found: {playbook_path}"
        
        scanner = AnsibleScanner()
        scanner.silent = True
        scanner.run(target_dir=full_path)
        
        # Get task information
        output = {
            "playbook": playbook_path,
            "status": "analyzed",
            "details": "Task extraction completed"
        }
        
        return json.dumps(output, indent=2)
    except ImportError:
        return "Error: ansible-content-capture is not properly installed."
    except Exception as e:
        return f"Error extracting playbook tasks: {str(e)}"

@tool
def list_ansible_modules(search_path: str = "") -> str:
    """List all Ansible modules used in the repository with usage count.
    
    Args:
        search_path: Path to search for modules (default: entire repo)
    
    Returns:
        List of modules with usage count and locations
    """
    try:
        from ansible_content_capture.scanner import AnsibleScanner
        
        scan_path = os.path.join(REPO_LOCAL_PATH, search_path) if search_path else REPO_LOCAL_PATH
        
        scanner = AnsibleScanner()
        scanner.silent = True
        scanner.run(target_dir=scan_path)
        
        output = {
            "search_path": search_path or "entire repository",
            "status": "Module scan completed",
            "note": "Use grep_search to find specific module usage patterns"
        }
        
        return json.dumps(output, indent=2)
    except ImportError:
        return "Error: ansible-content-capture is not properly installed."
    except Exception as e:
        return f"Error listing modules: {str(e)}"

@tool  
def extract_ansible_variables(content_path: str) -> str:
    """Extract all variables defined and used in Ansible playbooks/roles.
    
    Args:
        content_path: Path to playbook, role, or directory to analyze
    
    Returns:
        Dictionary of variables with their sources and usage locations
    """
    try:
        from ansible_content_capture.scanner import AnsibleScanner
        
        full_path = os.path.join(REPO_LOCAL_PATH, content_path)
        
        if not os.path.exists(full_path):
            return f"Error: Path not found: {content_path}"
        
        scanner = AnsibleScanner()
        scanner.silent = True
        scanner.run(target_dir=full_path)
        
        output = {
            "analyzed_path": content_path,
            "status": "Variable extraction completed",
            "note": "Variables have been analyzed from the specified content"
        }
        
        return json.dumps(output, indent=2)
    except ImportError:
        return "Error: ansible-content-capture is not properly installed."
    except Exception as e:
        return f"Error extracting variables: {str(e)}"

@tool
def analyze_role_structure(role_path: str) -> str:
    """Analyze the structure of an Ansible role including tasks, handlers, vars, and dependencies.
    
    Args:
        role_path: Path to the role directory relative to repo root
    
    Returns:
        JSON with role structure, components, and dependencies
    """
    try:
        from ansible_content_capture.scanner import AnsibleScanner
        
        full_path = os.path.join(REPO_LOCAL_PATH, role_path)
        
        if not os.path.exists(full_path):
            return f"Error: Role directory not found: {role_path}"
        
        # Check if it's a valid role directory
        role_components = ["tasks", "handlers", "defaults", "vars", "meta", "templates", "files"]
        existing_components = [comp for comp in role_components if os.path.exists(os.path.join(full_path, comp))]
        
        scanner = AnsibleScanner()
        scanner.silent = True
        scanner.run(target_dir=full_path)
        
        output = {
            "role_path": role_path,
            "components_found": existing_components,
            "status": "Role structure analyzed",
            "component_count": len(existing_components)
        }
        
        return json.dumps(output, indent=2)
    except ImportError:
        return "Error: ansible-content-capture is not properly installed."
    except Exception as e:
        return f"Error analyzing role: {str(e)}"

# Chain of Thought prompt for planning
COT_PLANNING_PROMPT = """You are an expert Ansible coding assistant. Before taking any action, think through the problem step by step.

For the user's query, create a detailed plan following this Chain of Thought approach:

STEP 1 - UNDERSTAND THE QUERY:
- What is the user asking for?
- What information do I need to answer this?
- Is this a question, modification request, or analysis task?

STEP 2 - PLAN THE APPROACH:
- Which tools should I use and in what order?
- Start with discovery (analyze_ansible_structure, find_relevant_files)
- Then narrow down (grep_search, search_in_files)
- MANDATORY: If grep_search finds NO results → immediately use intelligent_search with same term
- Finally read specific content (get_file_summary, read_file)
- For modifications: ALWAYS plan THREE steps:
  1) create_modification_plan (for approval)
  2) execute_modification_plan (to apply changes)
  3) verify_modification (to confirm success and retry if needed)

STEP 3 - ESTIMATE SCOPE:
- How many files will I likely need to examine?
- Can I answer with search results alone, or do I need to read files?
- For large files, plan to use line ranges

STEP 4 - EXECUTE EFFICIENTLY:
- Use the minimum number of tools necessary
- Read only relevant files/sections
- Limit file reads to 3-5 files maximum unless critical

CRITICAL ANTI-HALLUCINATION RULES (MANDATORY - FOLLOW EXACTLY):

Rule 1: ALWAYS USE ABSOLUTE PATHS FROM SEARCH TOOLS
  - grep_search and intelligent_search return ABSOLUTE paths
  - ABSOLUTE paths start with / (e.g., /Users/user/project/file.yml)
  - NEVER modify, shorten, or add to these absolute paths
  - COPY the EXACT absolute path from search results

Rule 2: EXTRACT ABSOLUTE PATHS FROM SEARCH RESULTS
  - Search results format: '/absolute/path/to/file.yml:LINE_NUMBER: content'
  - The file path is EVERYTHING before the FIRST colon
  - Example: '/Users/user/RHEL8-CIS/vars/main.yml:4: content' 
    → file path is '/Users/user/RHEL8-CIS/vars/main.yml'
  - NEVER truncate or modify the path

Rule 3: USE ABSOLUTE PATHS EXACTLY AS SHOWN
  - intelligent_search shows >>> /absolute/path/to/file <<<
  - COPY that ENTIRE absolute path EXACTLY
  - Do NOT remove any part of the path
  - Do NOT add 'roles/', './', or any prefix/suffix

Rule 4: IF read_file FAILS WITH 'does not exist'
  - You modified the path - THIS IS WRONG
  - Go back to search results
  - Find the >>> /absolute/path <<< section
  - Copy the COMPLETE path starting with /
  - Use that EXACT path with read_file()

Rule 5: NEVER GUESS OR CONSTRUCT PATHS
  - Do NOT assume directory structure
  - Do NOT construct paths like 'roles/NAME/vars/main.yml'
  - ONLY use absolute paths from search tool output
  - If search shows /Users/.../vars/main.yml, use that EXACTLY

Rule 6: VERIFICATION LOOP
  - Extract absolute path (everything before first colon)
  - Call read_file with COMPLETE absolute path
  - If it fails, you truncated/modified the path - go back to search results

Rule 7: DISCOVERED PATHS ARE PROVIDED IN EXECUTION CONTEXT
  - When executing steps, discovered file paths are shown in the prompt
  - Look for "DISCOVERED FILE PATHS" section in the execution context
  - ONLY use paths from this list - these are the ONLY valid paths
  - If you need a file path for a tool argument, copy it EXACTLY from this list
  - Do NOT generate, construct, or modify paths - ONLY use discovered paths

MANDATORY WORKFLOW WHEN USING intelligent_search:
Step 1: Call intelligent_search and receive results
Step 2: Look for the section with "!!!!..." and ">>> path <<<"
Step 3: In your thinking, explicitly write: "EXTRACTED PATH: [paste the complete path]"
Step 4: Verify the path starts with / (forward slash)
Step 5: Call read_file with that EXACT path (do NOT modify it)
Step 6: If read_file fails, you used the wrong path - go back to Step 2

CRITICAL: Before calling read_file, you MUST:
1. Find the line that starts with ">>>"
2. Extract everything between ">>> " and " <<<"
3. That is your file_path parameter
4. DO NOT construct paths like "roles/something/file.yml"
5. DO NOT use relative paths
6. ONLY use the absolute path from >>> <<< markers

EXAMPLE (FOLLOW THIS PATTERN):
If intelligent_search shows:
  >>> /Users/user/github_agent/RHEL8-CIS/vars/main.yml <<<
Then you MUST call:
  read_file('/Users/user/github_agent/RHEL8-CIS/vars/main.yml')
NOT:
  read_file('vars/main.yml')  [WRONG - truncated absolute path]
  read_file('roles/rhel8-cis/vars/main.yml')  [WRONG - hallucinated path]
  read_file('/Users/user/RHEL8-CIS/vars/main.yml')  [WRONG - removed part of path]

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
STOP! READ THIS BEFORE EVERY read_file() CALL:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

When you see intelligent_search results showing:
  >>> /Users/balasriharshach/BalaSriharsha/github_agent/RHEL8-CIS/vars/main.yml <<<

You MUST use that EXACT path:
  ✓ CORRECT: read_file('/Users/balasriharshach/BalaSriharsha/github_agent/RHEL8-CIS/vars/main.yml')
  ✗ WRONG:   read_file('roles/rhel8-cis/vars/main.yml')
  ✗ WRONG:   read_file('vars/main.yml')

NEVER construct paths like 'roles/something/vars/file.yml'
NEVER use relative paths
ALWAYS use the COMPLETE absolute path from >>> <<< markers

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Available Tools:

Ansible Tools:
- analyze_ansible_structure: Repository overview
- find_relevant_files: Find relevant files by keywords
- get_file_summary: Preview file without full read
- list_files: List directory contents
- grep_search: Regex search (returns ABSOLUTE paths in results)
- intelligent_search: Smart search that tries naming variations (returns ABSOLUTE paths - use when grep_search finds nothing)
- read_all_found_files: Automatically reads ALL files from last search (use this after intelligent_search instead of manually calling read_file)
- search_in_files: Simple text search (returns matching lines only)
- read_file: Read file using ABSOLUTE path from search results (paths start with /)
- verify_modification: MANDATORY tool to verify changes after execution (enables retry loop)

CRITICAL SEARCH STRATEGY (MANDATORY):
When searching for variables/config names, follow this EXACT sequence:

1. First: Try grep_search(pattern, file_pattern)
2. If "No matches found" → IMMEDIATELY call intelligent_search(pattern, file_pattern)
3. After intelligent_search succeeds → IMMEDIATELY call read_all_found_files()
   - This automatically reads ALL files found in the search
   - You see the complete content of each file
   - You can decide which file to modify based on actual content
4. intelligent_search will automatically try 7+ naming variations:
   - original: abc_def_ghi
   - no underscores: abcdefghi
   - combined prefix: abcdef_ghi (THIS OFTEN WORKS!)
   - camelCase, hyphenated, flexible regex, etc.
5. Review the file contents from read_all_found_files()
6. Select the correct file based on content (not just filename)
7. Use the ABSOLUTE path from the file listing for modifications

NEVER manually call read_file() - ALWAYS use read_all_found_files() after search!
NEVER stop after just grep_search - ALWAYS try intelligent_search if no results found!

Modification Tools (MUST follow this EXACT workflow):
Step 1: create_modification_plan - Prepare modification plan (stores plan, does NOT request approval yet)
Step 2: execute_modification_plan - Show plan to user, request approval, and execute changes (ATOMIC operation)
Step 3: verify_modification - MANDATORY verification after execution (see Self-Correction Loop below)

SELF-CORRECTION LOOP (CRITICAL):
After execute_modification_plan succeeds, you MUST:
1. Call verify_modification(file_path, expected_description, search_pattern)
2. If verification returns SUCCESS → Task complete!
3. If verification returns FAILED → You MUST retry:
   a. Read the file to see current state
   b. Create a NEW modification plan with corrected parameters
   c. Execute the new plan
   d. Verify again
   e. Repeat until verification succeeds

NEVER skip verification! NEVER stop after just one failed attempt!

Git Tools:
- git_fetch_all: Fetch latest changes from remote (automatically done at start)
- git_get_current_branch: Get current branch name
- git_get_base_branch: Determine base/parent branch
- git_sync_with_base: Merge base branch into current branch
- git_create_branch: Create a new branch and switch to it
- git_get_status: Get repository status (modified/staged files)
- git_list_branches: List all local branches

Ansible Content Analysis Tools:
- scan_ansible_content: Scan and analyze Ansible content (playbooks, roles, tasks) for detailed information
- extract_playbook_tasks: Extract tasks and execution flow from a specific playbook
- list_ansible_modules: List all Ansible modules used in the repository
- extract_ansible_variables: Extract variables defined and used in playbooks/roles
- analyze_role_structure: Analyze role structure including tasks, handlers, vars, dependencies

CRITICAL MODIFICATION WORKFLOW - SURGICAL EDITS ONLY!

ALWAYS USE SURGICAL EDITS (old_string/new_string) - NEVER rewrite entire files!

When ANY file modification is requested:

Step 1: Call intelligent_search to find the file
Step 2: Call read_all_found_files() to read the ACTUAL file content
Step 3: Review the ACTUAL content shown in the output - DO NOT SKIP THIS!
Step 4: Copy the EXACT string from the actual file that you want to replace (old_string)
Step 5: Create the MINIMAL replacement (new_string) with only what needs to change
Step 6: Call create_modification_plan with old_string and new_string parameters
Step 7: IMMEDIATELY call execute_modification_plan (shows plan, requests approval, executes)

IMPORTANT - SURGICAL EDIT RULES:
ALWAYS use old_string and new_string parameters in create_modification_plan
NEVER generate full file content - only the minimal string replacement
Make old_string specific enough to match exactly once in the file
Only include surrounding context if needed to make the match unique
Change only what's necessary - like adding a single line to a list

CRITICAL ANTI-CONTENT-HALLUCINATION RULES:
1. ALWAYS read the file with read_all_found_files() BEFORE creating a modification plan
2. COPY the exact text from the actual file output - do NOT guess or generate content
3. The old_string MUST be EXACTLY as it appears in the file (spacing, quotes, indentation)
4. If you guess the old_string without reading the file, the modification will FAIL
5. The system will validate that old_string exists in the file - hallucinated content will be REJECTED

THE APPROVAL WORKFLOW:
- create_modification_plan: Prepares surgical edit (NO user interaction) - USE ABSOLUTE PATHS
- execute_modification_plan: Shows diff → asks for branch → requests approval → executes

CRITICAL: When creating modification plan, use ABSOLUTE path from search results!
Example: file_path='/Users/user/github_agent/RHEL8-CIS/vars/main.yml'
NOT: file_path='roles/rhel8-cis/vars/main.yml' [WRONG]
NOT: file_path='vars/main.yml' [WRONG]

YOU MUST CALL BOTH TOOLS IN SEQUENCE:
1. create_modification_plan with old_string/new_string AND absolute file_path (prepares the edit)
2. execute_modification_plan (handles approval and execution)

If you only call create_modification_plan, the file is NOT modified!

NOTE: Git fetch and sync happen automatically at the start of each query.

Think step by step and create a clear plan before using tools."""

# System prompt for tool execution
SYSTEM_PROMPT = """You are executing a plan to help with Ansible code.

Execute the planned steps efficiently:
- Use tools as planned
- Minimize file reads
- Focus on relevant information
- Provide clear, concise responses

CRITICAL WORKFLOW AFTER intelligent_search (MANDATORY):

Step 1: intelligent_search finds files and shows:
```
>>> /Users/balasriharshach/BalaSriharsha/github_agent/RHEL8-CIS/vars/main.yml <<<
>>> /Users/balasriharshach/BalaSriharsha/github_agent/RHEL8-CIS/tasks/main.yml <<<
```

Step 2: IMMEDIATELY call read_all_found_files()
- This automatically reads ALL files found by the search
- No need to call read_file() manually
- You see complete content of each file
- Decide which file to modify based on actual content

Step 3: Create modification plan with ABSOLUTE path from file listing
✓ CORRECT: create_modification_plan(file_path='/Users/balasriharshach/BalaSriharsha/github_agent/RHEL8-CIS/vars/main.yml', ...)
✗ WRONG:   create_modification_plan(file_path='roles/rhel8-cis/vars/main.yml', ...)
✗ WRONG:   create_modification_plan(file_path='vars/main.yml', ...)

The path starts with / and contains the full directory structure. Use it EXACTLY as shown.

MANDATORY FILE MODIFICATION RULES:

RULE #1: ALWAYS USE SURGICAL EDITS
   - ALWAYS provide old_string and new_string parameters to create_modification_plan
   - NEVER generate complete file content - only the minimal replacement needed
   - Make targeted, minimal changes like Cursor, Windsurf, or Claude would
   - Example: To add one line, only replace the relevant section, not the entire file

RULE #2: TWO-STEP WORKFLOW (both tools required)
   a) create_modification_plan with old_string/new_string AND absolute file_path (prepares surgical edit)
   b) execute_modification_plan (shows diff, requests approval, executes)

RULE #3: ALWAYS USE ABSOLUTE PATHS FROM SEARCH RESULTS
   - intelligent_search shows: >>> /Users/.../RHEL8-CIS/vars/main.yml <<<
   - Extract that EXACT path and use it with read_file() and create_modification_plan()
   - NEVER construct paths like 'roles/something/vars/main.yml'
   - NEVER use relative paths like 'vars/main.yml'
   - ONLY use COMPLETE absolute paths from >>> <<< markers

RULE #4: Approval happens INSIDE execute_modification_plan
   - Shows user a clean diff of only what's changing
   - User sees: "- old line" and "+ new line"
   - Approval happens EXACTLY ONCE, right before execution

RULE #5: The file is NOT modified until execute_modification_plan completes

RULE #6: NEVER use write_file - it is DEPRECATED and bypasses approval

RULE #7: NEVER claim changes are complete after only calling create_modification_plan

RULE #8: If old_string appears multiple times, make it more specific

Git workflow:
- Git fetch and sync happen automatically at the start of each query
- Users can create feature/bugfix/chore/hotfix branches for modifications
- Branch creation is optional - users can work on current branch if preferred

When you have enough information, provide your answer."""

# State definition for Chain of Thought agent
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    plan_steps: list
    current_step: int
    step_results: list
    tools_used: int

# Create tools list
TOOLS = [
    analyze_ansible_structure,
    find_relevant_files,
    get_file_summary,
    list_files,
    grep_search,
    intelligent_search,
    read_all_found_files,
    search_in_files,
    read_file,
    create_modification_plan,
    verify_modification,
    write_file,
    execute_modification_plan,
    git_fetch_all,
    git_get_current_branch,
    git_get_base_branch,
    git_sync_with_base,
    git_create_branch,
    git_get_status,
    git_list_branches,
    scan_ansible_content,
    extract_playbook_tasks,
    list_ansible_modules,
    extract_ansible_variables,
    analyze_role_structure,
]

# Initialize the Chain of Thought agent
def create_ansible_agent():
    """Create and return the Ansible Chain of Thought agent."""
    # Use lower temperature for more focused, context-based responses
    # llm = VegasChatLLM(
    #     prompt_id = "ANSIBLE_AGENT_PROMPT"
    # )

    # Configure LLM with Google Gemini
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
        convert_system_message_to_human=True
    )
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Node 1: Planning with Chain of Thought
    def plan_step(state: AgentState):
        """First step: Create a detailed plan with numbered steps."""
        messages = state["messages"]
        user_query = messages[-1].content if messages else ""
        
        # Git sync workflow
        try:
            repo = Repo(REPO_LOCAL_PATH)
            
            # 1. Fetch latest from remote
            print_thinking("Fetching latest changes from remote...", "GIT")
            try:
                repo.remotes.origin.fetch()
                print_thinking("Fetch complete", "GIT")
            except Exception as e:
                print_thinking(f"Could not fetch from remote: {str(e)}", "GIT")
            
            # 2. Get current branch
            try:
                current_branch = repo.active_branch.name
                print_thinking(f"Current branch: {current_branch}", "GIT")
            except Exception as e:
                print_thinking(f"Could not determine current branch: {str(e)}", "GIT")
                current_branch = None
            
            # 3. Get base branch
            base_branch = None
            if current_branch:
                try:
                    active = repo.active_branch
                    if active.tracking_branch():
                        tracking = active.tracking_branch().name
                        base_branch = tracking.split('/')[-1] if '/' in tracking else tracking
                    else:
                        # Fallback to main or master
                        branch_names = [b.name for b in repo.heads]
                        if 'main' in branch_names:
                            base_branch = 'main'
                        elif 'master' in branch_names:
                            base_branch = 'master'
                    
                    if base_branch:
                        print_thinking(f"Base branch: {base_branch}", "GIT")
                except Exception as e:
                    print_thinking(f"Could not determine base branch: {str(e)}", "GIT")
            
            # 4. Sync with base branch (only if not on base branch itself)
            if base_branch and current_branch and current_branch != base_branch:
                try:
                    print_thinking(f"Syncing {current_branch} with {base_branch}...", "GIT")
                    repo.git.merge(base_branch, '--no-edit')
                    print_thinking("Sync complete", "GIT")
                except Exception as e:
                    if "CONFLICT" in str(e):
                        print_thinking("Merge conflict detected. Manual resolution required.", "GIT")
                    else:
                        print_thinking(f"Could not sync with base: {str(e)}", "GIT")
        except Exception as e:
            print_thinking(f"Git operations skipped: {str(e)}", "GIT")
        
        print_thinking("Analyzing user request...", "THINKING")
        print_thinking(f"User Query: {user_query}", "THINKING")
        print_thinking("Creating detailed execution plan...", "PLANNING")
        
        # Enhanced planning prompt that asks for numbered steps
        planning_message = HumanMessage(content=f"""{COT_PLANNING_PROMPT}

User Query: {user_query}

Now, create a detailed execution plan. Format your plan as a numbered list of specific steps:

EXECUTION PLAN:
Step 1: [First action to take]
Step 2: [Second action to take]
...

Be specific about which tools to use in each step.""")
        
        # Get the plan from LLM
        response = llm.invoke([planning_message])
        
        print_thinking("LLM generated execution plan", "MODEL")
        print_thinking("Plan generation complete. Parsing steps...", "PLANNING")
        
        # Parse the plan into individual steps
        plan_text = response.content
        steps = []
        
        # Extract numbered steps from the plan
        import re
        step_pattern = r'Step \d+:(.+?)(?=Step \d+:|$)'
        matches = re.findall(step_pattern, plan_text, re.DOTALL)
        
        if matches:
            steps = [step.strip() for step in matches]
        else:
            # Fallback: split by newlines if no numbered steps found
            lines = [line.strip() for line in plan_text.split('\n') if line.strip() and not line.strip().startswith('#')]
            steps = [line for line in lines if len(line) > 10][:Config.MAX_PLAN_STEPS]
        
        print_thinking(f"Extracted {len(steps)} execution steps", "PLANNING")
        print_section("EXECUTION PLAN")
        print_thinking("Plan details:", "PLAN")
        # Print plan text line by line to keep it in thinking context
        for line in plan_text.split('\n'):
            if line.strip():
                print(line)
        
        return {
            "messages": [AIMessage(content=f"[PLAN CREATED]\n{response.content}")],
            "plan": response.content,
            "plan_steps": steps,
            "current_step": 0,
            "step_results": [],
            "tools_used": 0
        }
    
    # Node 2: Review the plan
    def review_plan(state: AgentState):
        """Review the plan before execution."""
        plan = state["plan"]
        
        print_thinking(f"Reviewing plan with {len(state['plan_steps'])} steps", "REVIEW")
        print_thinking("Plan looks good. Ready to execute.", "REVIEW")
        print_thinking("Starting execution...", "EXECUTION")
        
        review_message = AIMessage(content=f"[PLAN REVIEW]\n\nPlan has {len(state['plan_steps'])} steps. Ready to execute.\n\nPlan:\n{plan}\n\n[Starting execution...]")
        
        return {
            "messages": [review_message]
        }
    
    # Node 3: Execute one step at a time
    def execute_step(state: AgentState):
        """Execute the current step from the plan."""
        current_step_idx = state["current_step"]
        plan_steps = state["plan_steps"]
        
        # Check if we have more steps to execute
        if current_step_idx >= len(plan_steps):
            print_thinking("All steps completed", "EXECUTION")
            return {"messages": [AIMessage(content="[All steps completed]")]}
        
        current_step_text = plan_steps[current_step_idx]
        original_query = state["messages"][0].content
        
        print_section(f"EXECUTING STEP {current_step_idx + 1}/{len(plan_steps)}")
        print_thinking(f"Step: {current_step_text}", "STEP")
        print_thinking("Determining which tools to use...", "THINKING")
        
        # Detect if tools are mentioned in the step
        tool_names = [tool.name for tool in TOOLS]
        mentioned_tools = [name for name in tool_names if name in current_step_text.lower().replace('_', ' ') or name in current_step_text]
        
        # Get discovered file paths from global state
        global _LAST_SEARCH_PATHS
        discovered_paths_context = ""
        if _LAST_SEARCH_PATHS:
            paths_list = "\n".join(f"  - {p}" for p in _LAST_SEARCH_PATHS)
            discovered_paths_context = f"""

DISCOVERED FILE PATHS (use ONLY these paths - DO NOT generate new paths):
{paths_list}

CRITICAL: If you need to use a file path in tool arguments, you MUST copy one of the paths listed above EXACTLY.
DO NOT construct, generate, or guess file paths. ONLY use paths from the list above.
"""
        
        # Get file content from previous steps if read_all_found_files was called
        file_content_context = ""
        for result in state['step_results'][-5:]:
            if "FILE:" in str(result) and "=" * 80 in str(result):
                # This looks like output from read_all_found_files
                file_content_context = f"""

ACTUAL FILE CONTENT FROM PREVIOUS STEP:
{str(result)[:2000]}

CRITICAL: Use the EXACT content shown above when creating modifications!
DO NOT generate or guess file content - COPY from the output above!
"""
                break
        
        # Create execution prompt with strong tool enforcement
        if mentioned_tools:
            tool_list = ", ".join(mentioned_tools)
            step_prompt = HumanMessage(content=f"""EXECUTE THIS STEP EXACTLY AS DESCRIBED.

Original Query: {original_query}

Current Step ({current_step_idx + 1}/{len(plan_steps)}): {current_step_text}

Previous Results: {state['step_results'][-3:] if state['step_results'] else 'None'}
{discovered_paths_context}
{file_content_context}

CRITICAL: This step mentions these tools: {tool_list}
You MUST call the tools mentioned in the step description.
If the step says "call create_modification_plan", you MUST call create_modification_plan.
If the step says "call execute_modification_plan", you MUST call execute_modification_plan.
DO NOT say "No tools needed" - the step explicitly requires tool calls.

IMPORTANT: If creating a modification, use the EXACT content from the file output above.
DO NOT guess or generate content that isn't in the actual file!

RETRY LOGIC:
- If a tool call returns an error (e.g., "CONTENT HALLUCINATION DETECTED"), you MUST retry in THIS SAME STEP
- You can make up to 3 tool calls in one step - use them to retry until success
- Do NOT move to next step if you get an error - FIX IT NOW!

Execute this step NOW by calling the appropriate tools.""")
        else:
            step_prompt = HumanMessage(content=f"""Execute this step from the plan. Be CONCISE in your response.

Original Query: {original_query}

Current Step ({current_step_idx + 1}/{len(plan_steps)}): {current_step_text}

Previous Results: {state['step_results'][-3:] if state['step_results'] else 'None'}
{discovered_paths_context}
{file_content_context}

IMPORTANT: If creating a modification, use the EXACT content from the file output above.
DO NOT guess or generate content that isn't in the actual file!

RETRY LOGIC:
- If a tool call returns an error (e.g., "CONTENT HALLUCINATION DETECTED"), you MUST retry in THIS SAME STEP
- You can make up to 3 tool calls in one step - use them to retry until success
- Do NOT move to next step if you get an error - FIX IT NOW!

Execute this step now. Use only the necessary tools. Keep your response brief and focused.""")
        
        # Let LLM execute the step with tools
        response = llm_with_tools.invoke([step_prompt])
        
        # Display LLM reasoning if present
        if hasattr(response, 'content') and response.content and response.content.strip():
            print_thinking(f"LLM Reasoning: {response.content[:300]}", "MODEL")
        
        # Check if tools were called
        tool_messages = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print_thinking(f"Calling {len(response.tool_calls)} tool(s)...", "EXECUTION")
            # Execute tool calls (limit to 3 per step)
            for tool_call in response.tool_calls[:3]:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print_thinking(f"Tool: {tool_name}", "TOOL")
                print_thinking(f"Arguments: {str(tool_args)[:Config.TOOL_OUTPUT_PREVIEW_CHARS]}", "TOOL")
                
                for tool in TOOLS:
                    if tool.name == tool_name:
                        try:
                            result = tool.invoke(tool_args)
                            
                            # For critical tools, show full output (no truncation)
                            # For other tools, show reasonable preview
                            critical_tools = [
                                'intelligent_search', 'grep_search', 'search_in_files',
                                'read_all_found_files', 'read_file', 
                                'create_modification_plan', 'apply_file_edit',
                                'execute_modification_plan', 'verify_modification'
                            ]
                            
                            if tool_name in critical_tools:
                                # Show full output for critical tools - NO TRUNCATION
                                print_thinking(f"Result: {str(result)}", "TOOL")
                            else:
                                # Show preview for non-critical tools only
                                result_preview = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
                                print_thinking(f"Result: {result_preview}", "TOOL")
                            
                            tool_messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"]
                            ))
                        except Exception as e:
                            print_thinking(f"Error: {str(e)}", "TOOL")
                            tool_messages.append(ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call["id"]
                            ))
                        break
        else:
            print_thinking("No tools needed for this step", "EXECUTION")
        
        print_thinking(f"Step {current_step_idx + 1} complete", "EXECUTION")
        
        # Store step result - DO NOT TRUNCATE for critical tools
        step_result = {
            "step": current_step_idx + 1,
            "action": current_step_text,
            "tools_used": [msg.content for msg in tool_messages]  # Full content, no truncation
        }
        
        new_step_results = state["step_results"] + [step_result]
        
        return {
            "messages": [response] + tool_messages,
            "current_step": current_step_idx + 1,
            "step_results": new_step_results,
            "tools_used": state["tools_used"] + len(tool_messages)
        }
    
    # Node 4: Generate final answer
    def generate_answer(state: AgentState):
        """Generate final answer based on gathered information."""
        original_query = state["messages"][0].content
        step_results = state["step_results"]
        
        print_thinking("Synthesizing final answer...", "SYNTHESIS")
        print_thinking("Analyzing results from all executed steps...", "THINKING")
        print_thinking(f"Executed {len(step_results)} steps total", "THINKING")
        
        # Create a summary of what was done
        summary = "\n".join([f"Step {r['step']}: {r['action']}" for r in step_results])
        
        # Check if this was a modification request
        modification_keywords = ['add', 'modify', 'change', 'update', 'remove', 'delete', 'create', 'write']
        is_modification_request = any(keyword in original_query.lower() for keyword in modification_keywords)
        
        # Check if both modification tools were called
        messages = state["messages"]
        modification_plan_called = any(
            hasattr(msg, 'tool_calls') and msg.tool_calls and 
            any(tc.get('name') == 'create_modification_plan' for tc in msg.tool_calls)
            for msg in messages if hasattr(msg, 'tool_calls')
        )
        
        execute_plan_called = any(
            hasattr(msg, 'tool_calls') and msg.tool_calls and 
            any(tc.get('name') == 'execute_modification_plan' for tc in msg.tool_calls)
            for msg in messages if hasattr(msg, 'tool_calls')
        )
        
        print_thinking("Creating final response for user...", "THINKING")
        
        # Get discovered file paths for context
        global _LAST_SEARCH_PATHS
        discovered_paths_info = ""
        if _LAST_SEARCH_PATHS:
            paths_list = "\n".join(f"  - {p}" for p in _LAST_SEARCH_PATHS)
            discovered_paths_info = f"""

DISCOVERED FILE PATHS (DO NOT hallucinate paths - ONLY use these):
{paths_list}

CRITICAL: If you mention file paths in your answer, ONLY use the exact paths listed above.
DO NOT construct, generate, or guess file paths like 'roles/something/file.yml'.
"""
        
        # Build the final prompt with modification check
        if is_modification_request and not modification_plan_called:
            final_prompt = HumanMessage(content=f"""Based on the plan execution, provide a CONCISE answer to the user's query.

Original Query: {original_query}

Steps Executed:
{summary}
{discovered_paths_info}

IMPORTANT: This was a file modification request, but create_modification_plan was NOT called during execution.
You MUST tell the user that the modification was NOT completed because the approval workflow was not followed.
Explain that they need to run the request again and ensure BOTH create_modification_plan AND execute_modification_plan are called.

If mentioning file paths, use ONLY the discovered paths listed above.""")
        elif is_modification_request and modification_plan_called and not execute_plan_called:
            final_prompt = HumanMessage(content=f"""Based on the plan execution, provide a CONCISE answer to the user's query.

Original Query: {original_query}

Steps Executed:
{summary}
{discovered_paths_info}

CRITICAL: The modification was NOT completed! 
While create_modification_plan was called and approved, execute_modification_plan was NEVER called.
The file was NOT actually modified.
You MUST tell the user that the changes were NOT applied because execute_modification_plan was not called.
The workflow requires BOTH tools: create_modification_plan (approval) AND execute_modification_plan (actual modification).

If mentioning file paths, use ONLY the discovered paths listed above.""")
        else:
            final_prompt = HumanMessage(content=f"""Based on the plan execution, provide a CONCISE answer to the user's query.

Original Query: {original_query}

Steps Executed:
{summary}
{discovered_paths_info}

Provide a clear, brief answer focusing only on what the user asked.

If mentioning file paths, use ONLY the discovered paths listed above - DO NOT hallucinate paths.""")
        
        response = llm.invoke([final_prompt])
        
        # Display any intermediate LLM reasoning
        if hasattr(response, 'content') and response.content:
            print_thinking("LLM generated response", "MODEL")
        
        print_thinking("Answer ready", "COMPLETE")
        print_thinking("Preparing final response...", "COMPLETE")
        
        return {"messages": [response]}
    
    # Conditional edge: Continue executing steps or generate answer
    def should_continue_steps(state: AgentState):
        """Decide whether to execute more steps or generate the final answer."""
        current_step = state["current_step"]
        total_steps = len(state["plan_steps"])
        
        # If we have more steps to execute and haven't used too many tools
        if current_step < total_steps and state["tools_used"] < 15:
            return "continue"
        
        # Otherwise, generate answer
        return "answer"
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_step)
    workflow.add_node("review", review_plan)
    workflow.add_node("execute_step", execute_step)
    workflow.add_node("answer", generate_answer)
    
    # Add edges
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "review")
    workflow.add_edge("review", "execute_step")
    workflow.add_conditional_edges(
        "execute_step",
        should_continue_steps,
        {
            "continue": "execute_step",
            "answer": "answer"
        }
    )
    workflow.add_edge("answer", END)
    
    # Compile the graph
    agent = workflow.compile()
    
    return agent

def main():
    """Main function to run the agent."""
    print("=== Ansible Chain of Thought Coding Agent ===\n")
    
    # Setup repository
    # if GIT_REPO_URL:
    #     setup_repo()
    # else:
    #     print("Warning: GIT_REPO_URL not set. Make sure ansible_repo directory exists.")
    
    # Create agent
    agent = create_ansible_agent()
    
    print("\nAgent ready. The agent will think step-by-step before answering.")
    print("Type your questions or requests (type 'quit' to exit):\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Start thinking tag for all processing
            start_thinking()
            
            print("\n" + "="*Config.SEPARATOR_WIDTH_STANDARD)
            print("AGENT PROCESSING")
            print("="*Config.SEPARATOR_WIDTH_STANDARD + "\n")
            
            # Invoke the Chain of Thought agent with initial state
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "plan": "",
                "plan_steps": [],
                "current_step": 0,
                "step_results": [],
                "tools_used": 0
            }
            
            result = agent.invoke(initial_state)
            
            # End thinking tag before final answer
            end_thinking()
            
            # Get the final message
            final_message = result["messages"][-1]
            
            # Display final answer outside thinking tag
            print("\n" + "="*Config.SEPARATOR_WIDTH_STANDARD)
            print("FINAL ANSWER")
            print("="*Config.SEPARATOR_WIDTH_STANDARD + "\n")
            
            if hasattr(final_message, 'content'):
                print(f"{final_message.content}\n")
            else:
                print(f"{str(final_message)}\n")
            
            print("="*Config.SEPARATOR_WIDTH_STANDARD + "\n")
                
        except Exception as e:
            # Make sure to close thinking tag on error
            end_thinking()
            print(f"\nError: {str(e)}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()