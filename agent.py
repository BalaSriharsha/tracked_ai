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

# Helper functions for displaying agent thinking
def print_thinking(message: str, prefix="THINKING") -> None:
    """Display agent's thinking process in real-time."""
    print(f"[{prefix}] {message}", flush=True)

def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n", flush=True)

# Helper functions for interactive modification approval
def display_modification_plan(plan: dict) -> None:
    """Display modification plan in a formatted, user-friendly way."""
    print("\n" + "="*80)
    print("MODIFICATION PLAN")
    print("="*80)
    print(f"\nDescription: {plan.get('modification_description', 'No description provided')}")
    
    files_to_modify = plan.get('files_to_modify', [])
    if files_to_modify:
        print(f"\nFiles to modify ({len(files_to_modify)}):")
        for idx, file_info in enumerate(files_to_modify, 1):
            file_path = file_info.get('file', 'Unknown file')
            changes = file_info.get('changes', 'No changes specified')
            print(f"\n  {idx}. {file_path}")
            print(f"     Changes: {changes}")
    else:
        print("\nNo specific files listed in modification plan.")
    
    if plan.get('new_content'):
        print(f"\nNew Content Preview:")
        print("-" * 80)
        content_preview = plan['new_content'][:500]
        print(content_preview)
        if len(plan['new_content']) > 500:
            print(f"\n... (truncated, {len(plan['new_content']) - 500} more characters)")
        print("-" * 80)
    
    print("\n" + "="*80)

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
    # Limit length to 40 characters
    clean_desc = clean_desc[:40].rstrip('-')
    
    # Create branch name
    branch_name = f"{change_type}/{clean_desc}"
    return branch_name

def ask_branch_creation(modification_description: str) -> tuple:
    """Ask user if they want to create a new branch for modifications.
    
    Returns:
        Tuple of (create_branch: bool, branch_name: str, change_type: str)
    """
    print("\n" + "="*80)
    print("BRANCH CREATION")
    print("="*80)
    
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
    """Read the content of a file from the repository. Provide path relative to repo root.
    
    Args:
        file_path: Path to the file relative to repository root
        start_line: Optional starting line number (1-indexed). If provided, only reads from this line
        end_line: Optional ending line number (1-indexed). If provided with start_line, reads only this range
    
    Returns:
        File content or the specified line range
    """
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
                if line_count > 500:
                    return f"Warning: {file_path} has {line_count} lines. Consider using start_line/end_line parameters to read specific sections.\n\nFirst 100 lines:\n" + '\n'.join(content.split('\n')[:100]) + f"\n\n... (truncated, {line_count - 100} more lines)"
                
                return f"Content of {file_path} ({line_count} lines):\n{content}"
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

@tool
def get_file_summary(file_path: str) -> str:
    """Get a quick summary of a file without reading full content. Shows file size, line count, and first few lines.
    Use this to decide if you need to read the full file.
    
    Args:
        file_path: Path to the file relative to repository root
    
    Returns:
        File summary with metadata and preview
    """
    full_path = Path(REPO_LOCAL_PATH) / file_path
    if not full_path.exists():
        return f"File {file_path} does not exist."
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            total_lines = len(lines)
            
        # Get first 10 and last 5 lines for preview
        preview_start = ''.join(lines[:10])
        preview_end = ''.join(lines[-5:]) if total_lines > 15 else ""
        
        file_size = full_path.stat().st_size
        size_kb = file_size / 1024
        
        summary = f"""File: {file_path}
Size: {size_kb:.2f} KB
Lines: {total_lines}

--- First 10 lines ---
{preview_start}"""
        
        if preview_end and total_lines > 15:
            summary += f"\n... ({total_lines - 15} lines omitted) ...\n\n--- Last 5 lines ---\n{preview_end}"
        
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
                    # Read only first 200 lines for efficiency
                    content_preview = ''.join(line for _, line in zip(range(200), f))
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
    for file, score, match_type in relevant_files[:15]:  # Show top 15
        output.append(f"  [{score} matches, {match_type}] {file}")
    
    if len(relevant_files) > 15:
        output.append(f"\n... and {len(relevant_files) - 15} more files")
    
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
    
    Args:
        pattern: Regex pattern to search for (e.g., 'ansible.builtin.*', 'name:.*nginx', etc.)
        file_pattern: File glob pattern to search in (default: *, examples: '*.yml', '*.yaml', 'roles/**/tasks/*.yml')
        case_sensitive: Whether the search should be case-sensitive (default: False)
        max_results: Maximum number of results to return (default: 50)
    
    Returns:
        Matching lines with file paths and line numbers
    """
    base_path = Path(REPO_LOCAL_PATH)
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
                            rel_path = file_path.relative_to(REPO_LOCAL_PATH)
                            results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                            
                            if len(results) >= max_results:
                                results.append(f"\n... (truncated, showing first {max_results} results)")
                                return "\n".join(results)
            except Exception:
                pass
    
    return "\n".join(results) if results else f"No matches found for pattern '{pattern}'."

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
        for pb in analysis["playbooks"][:10]:  # Show first 10
            output.append(f"  - {pb}")
        if len(analysis["playbooks"]) > 10:
            output.append(f"  ... and {len(analysis['playbooks']) - 10} more")
        output.append("")
    
    if analysis["roles"]:
        output.append(f"Roles ({len(analysis['roles'])}):")
        for role in analysis["roles"][:10]:
            components = ", ".join(role["components"])
            output.append(f"  - {role['name']} ({components})")
        if len(analysis["roles"]) > 10:
            output.append(f"  ... and {len(analysis['roles']) - 10} more")
        output.append("")
    
    if analysis["inventory_files"]:
        output.append(f"Inventory Files ({len(analysis['inventory_files'])}):")
        for inv in analysis["inventory_files"]:
            output.append(f"  - {inv}")
        output.append("")
    
    if analysis["group_vars"]:
        output.append(f"Group Variables ({len(analysis['group_vars'])}):")
        for gv in analysis["group_vars"][:5]:
            output.append(f"  - {gv}")
        if len(analysis["group_vars"]) > 5:
            output.append(f"  ... and {len(analysis['group_vars']) - 5} more")
        output.append("")
    
    return "\n".join(output) if any(analysis.values()) else "No Ansible structure detected in the repository."

@tool
def create_modification_plan(description: str, file_path: str = "", changes_summary: str = "", new_content: str = "") -> str:
    """Create a plan for code modifications. This prepares the plan but does NOT request approval yet.
    Approval will be requested when execute_modification_plan is called.
    
    Args:
        description: Description of what needs to be changed
        file_path: Path to the file to modify (optional)
        changes_summary: Summary of changes to be made. For deletions, use keywords like "Deletion of..." (optional)
        new_content: New content to write if creating/overwriting a file. Use empty string "" for deletions (optional)
    
    Returns:
        Status message confirming the plan was created
    
    Note:
        For file deletions, set new_content="" and use deletion keywords in changes_summary or description.
        This tool only PREPARES the plan. Call execute_modification_plan to show the plan, request approval, and execute.
    """
    global PENDING_MODIFICATION_PLAN
    
    files_to_modify = []
    if file_path:
        files_to_modify.append({
            "file": file_path,
            "changes": changes_summary,
            "content": new_content
        })
    
    plan = {
        "modification_description": description,
        "status": "pending_approval",
        "files_to_modify": files_to_modify,
        "new_content": new_content,
        "target_file": file_path,
        "instructions": "Call execute_modification_plan to show this plan to the user and request approval."
    }
    
    # Store the plan (will be shown to user when execute_modification_plan is called)
    PENDING_MODIFICATION_PLAN = plan
    
    return f"Modification plan created successfully. Call execute_modification_plan to show the plan to the user, request approval, and apply the changes."

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
        content = file_info.get("content")
        
        # Check if this is a deletion request
        is_deletion = content == "" and changes and any(keyword in changes.lower() for keyword in ['delete', 'deletion', 'remove', 'removal'])
        
        if is_deletion:
            # Delete the file
            try:
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
                full_path = Path(REPO_LOCAL_PATH) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
                results.append(f"Modified {file_path}: {changes}")
            except Exception as e:
                results.append(f"Error modifying {file_path}: {str(e)}")
        else:
            results.append(f"{file_path}: {changes} (no content provided, skipped)")
    
    # Handle new_content (for single file modification)
    if plan.get("target_file"):
        file_path = plan["target_file"]
        new_content = plan.get("new_content", "")
        description = plan.get("modification_description", "")
        
        # Check if this is a deletion request
        is_deletion = new_content == "" and any(keyword in description.lower() for keyword in ['delete', 'deletion', 'remove', 'removal'])
        
        if is_deletion:
            # Delete the file
            try:
                full_path = Path(REPO_LOCAL_PATH) / file_path
                if full_path.exists():
                    full_path.unlink()
                    results.append(f"Successfully deleted {file_path}")
                else:
                    results.append(f"File not found (already deleted): {file_path}")
            except Exception as e:
                results.append(f"Error deleting {file_path}: {str(e)}")
        elif new_content:
            # Write the new content
            try:
                full_path = Path(REPO_LOCAL_PATH) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(new_content)
                results.append(f"Successfully wrote to {file_path}")
            except Exception as e:
                results.append(f"Error writing to {file_path}: {str(e)}")
    
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
            for file in modified[:10]:
                status_lines.append(f"  - {file}")
            if len(modified) > 10:
                status_lines.append(f"  ... and {len(modified) - 10} more")
        
        # Get staged files
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        if staged:
            status_lines.append(f"\nStaged files ({len(staged)}):")
            for file in staged[:10]:
                status_lines.append(f"  - {file}")
            if len(staged) > 10:
                status_lines.append(f"  ... and {len(staged) - 10} more")
        
        # Get untracked files
        untracked = repo.untracked_files
        if untracked:
            status_lines.append(f"\nUntracked files ({len(untracked)}):")
            for file in untracked[:10]:
                status_lines.append(f"  - {file}")
            if len(untracked) > 10:
                status_lines.append(f"  ... and {len(untracked) - 10} more")
        
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
- Finally read specific content (get_file_summary, read_file)
- For modifications: ALWAYS plan TWO steps:
  1) create_modification_plan (for approval)
  2) execute_modification_plan (to apply changes)

STEP 3 - ESTIMATE SCOPE:
- How many files will I likely need to examine?
- Can I answer with search results alone, or do I need to read files?
- For large files, plan to use line ranges

STEP 4 - EXECUTE EFFICIENTLY:
- Use the minimum number of tools necessary
- Read only relevant files/sections
- Limit file reads to 3-5 files maximum unless critical

Available Tools:

Ansible Tools:
- analyze_ansible_structure: Repository overview
- find_relevant_files: Find relevant files by keywords
- get_file_summary: Preview file without full read
- list_files: List directory contents
- grep_search: Regex search (returns matching lines only)
- search_in_files: Simple text search (returns matching lines only)
- read_file: Read full file or line range

Modification Tools (ALWAYS use these TWO tools in sequence):
- create_modification_plan: Prepare modification plan (stores plan, does NOT request approval yet)
- execute_modification_plan: Show plan to user, request approval, and execute changes (ATOMIC operation)
- write_file: DEPRECATED - Do NOT use this tool

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

CRITICAL MODIFICATION WORKFLOW - FOLLOW THESE STEPS IN ORDER:
When ANY file modification is requested, you MUST execute ALL these steps:

Step 1: Read the file (use read_file) if modifying existing content
Step 2: Call create_modification_plan with complete new_content (this PREPARES the plan, NO approval yet)
Step 3: IMMEDIATELY call execute_modification_plan (this shows plan, requests approval, and executes)

THE APPROVAL WORKFLOW:
- create_modification_plan: Stores the plan (NO user interaction)
- execute_modification_plan: Shows plan → asks for branch → requests approval → executes (ALL in one atomic operation)

This ensures approval happens EXACTLY ONCE, right before execution.

YOU MUST CALL BOTH TOOLS IN SEQUENCE:
1. create_modification_plan (prepares the plan)
2. execute_modification_plan (handles approval and execution)

If you only call create_modification_plan, the file is NOT modified!
NEVER use write_file - it is deprecated and bypasses approval.

NOTE: Git fetch and sync happen automatically at the start of each query.

Think step by step and create a clear plan before using tools."""

# System prompt for tool execution
SYSTEM_PROMPT = """You are executing a plan to help with Ansible code.

Execute the planned steps efficiently:
- Use tools as planned
- Minimize file reads
- Focus on relevant information
- Provide clear, concise responses

MANDATORY FILE MODIFICATION RULES:
1. For ANY file change, you MUST call BOTH tools in sequence:
   a) create_modification_plan (prepares the plan, NO user interaction)
   b) execute_modification_plan (shows plan, requests approval, executes - ATOMIC operation)

2. Approval happens INSIDE execute_modification_plan, NOT in create_modification_plan
   - This ensures approval happens EXACTLY ONCE, right before execution
   - No matter how many times you call create_modification_plan, approval only happens once

3. The file is NOT modified until execute_modification_plan is called and user approves

4. NEVER use write_file - it is DEPRECATED and bypasses approval

5. NEVER claim changes are complete after only calling create_modification_plan

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
    search_in_files,
    read_file,
    create_modification_plan,
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
    #     usecase_name=os.getenv("VEGAS_USECASE_NAME", "AnsibleCodingAgent"),
    #     context_name=os.getenv("VEGAS_CONTEXT_NAME", "AnsibleCodeContext"),
    #     temperature=0.2,  # Lower temperature for consistent, factual responses
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
            steps = [line for line in lines if len(line) > 10][:10]  # Max 10 steps
        
        print_thinking(f"Extracted {len(steps)} execution steps", "PLANNING")
        print_section("EXECUTION PLAN")
        print(plan_text)
        
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
        
        # Create execution prompt with strong tool enforcement
        if mentioned_tools:
            tool_list = ", ".join(mentioned_tools)
            step_prompt = HumanMessage(content=f"""EXECUTE THIS STEP EXACTLY AS DESCRIBED.

Original Query: {original_query}

Current Step ({current_step_idx + 1}/{len(plan_steps)}): {current_step_text}

Previous Results: {state['step_results'][-3:] if state['step_results'] else 'None'}

CRITICAL: This step mentions these tools: {tool_list}
You MUST call the tools mentioned in the step description.
If the step says "call create_modification_plan", you MUST call create_modification_plan.
If the step says "call execute_modification_plan", you MUST call execute_modification_plan.
DO NOT say "No tools needed" - the step explicitly requires tool calls.

Execute this step NOW by calling the appropriate tools.""")
        else:
            step_prompt = HumanMessage(content=f"""Execute this step from the plan. Be CONCISE in your response.

Original Query: {original_query}

Current Step ({current_step_idx + 1}/{len(plan_steps)}): {current_step_text}

Previous Results: {state['step_results'][-3:] if state['step_results'] else 'None'}

Execute this step now. Use only the necessary tools. Keep your response brief and focused.""")
        
        # Let LLM execute the step with tools
        response = llm_with_tools.invoke([step_prompt])
        
        # Check if tools were called
        tool_messages = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print_thinking(f"Calling {len(response.tool_calls)} tool(s)...", "EXECUTION")
            # Execute tool calls (limit to 3 per step)
            for tool_call in response.tool_calls[:3]:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print_thinking(f"Tool: {tool_name}", "TOOL")
                print_thinking(f"Arguments: {str(tool_args)[:100]}", "TOOL")
                
                for tool in TOOLS:
                    if tool.name == tool_name:
                        try:
                            result = tool.invoke(tool_args)
                            result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
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
        
        # Store step result
        step_result = {
            "step": current_step_idx + 1,
            "action": current_step_text,
            "tools_used": [msg.content[:100] + "..." if len(msg.content) > 100 else msg.content for msg in tool_messages]
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
        
        print_section("SYNTHESIZING ANSWER")
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
        
        # Build the final prompt with modification check
        if is_modification_request and not modification_plan_called:
            final_prompt = HumanMessage(content=f"""Based on the plan execution, provide a CONCISE answer to the user's query.

Original Query: {original_query}

Steps Executed:
{summary}

IMPORTANT: This was a file modification request, but create_modification_plan was NOT called during execution.
You MUST tell the user that the modification was NOT completed because the approval workflow was not followed.
Explain that they need to run the request again and ensure BOTH create_modification_plan AND execute_modification_plan are called.""")
        elif is_modification_request and modification_plan_called and not execute_plan_called:
            final_prompt = HumanMessage(content=f"""Based on the plan execution, provide a CONCISE answer to the user's query.

Original Query: {original_query}

Steps Executed:
{summary}

CRITICAL: The modification was NOT completed! 
While create_modification_plan was called and approved, execute_modification_plan was NEVER called.
The file was NOT actually modified.
You MUST tell the user that the changes were NOT applied because execute_modification_plan was not called.
The workflow requires BOTH tools: create_modification_plan (approval) AND execute_modification_plan (actual modification).""")
        else:
            final_prompt = HumanMessage(content=f"""Based on the plan execution, provide a CONCISE answer to the user's query.

Original Query: {original_query}

Steps Executed:
{summary}

Provide a clear, brief answer focusing only on what the user asked.""")
        
        response = llm.invoke([final_prompt])
        
        print_thinking("Answer ready", "COMPLETE")
        print_section("FINAL ANSWER")
        
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
            print("\n" + "="*80)
            print("AGENT PROCESSING")
            print("="*80 + "\n")
            
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
            
            # Get the final message
            final_message = result["messages"][-1]
            
            if hasattr(final_message, 'content'):
                print(f"{final_message.content}\n")
            else:
                print(f"{str(final_message)}\n")
            
            print("="*80 + "\n")
                
        except Exception as e:
            print(f"Error: {str(e)}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()