import os
import json
import re
from pathlib import Path
from typing import List, Dict, Annotated, TypedDict
from dotenv import load_dotenv
from git import Repo
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pyvegas.langx.llm import VegasChatLLM

# Load environment variables
load_dotenv()

# Configuration
REPO_LOCAL_PATH = ".."
# GIT_REPO_URL = os.getenv("GIT_REPO_URL")

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
def create_modification_plan(description: str) -> str:
    """Create a plan for code modifications. Input should be a description of what needs to be changed.
    Returns a JSON plan with files to modify and proposed changes."""
    plan = {
        "modification_description": description,
        "status": "pending_approval",
        "files_to_modify": [],
        "instructions": "Review the plan and approve to proceed with modifications."
    }
    
    # Save plan for later execution
    with open("modification_plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    
    return f"Modification plan created. Review and add files to modify:\n{json.dumps(plan, indent=2)}\n\nTo proceed, use the execute_modification_plan tool after approval."

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the repository. This modifies the actual file."""
    full_path = Path(REPO_LOCAL_PATH) / file_path
    
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to {file_path}: {str(e)}"

@tool
def execute_modification_plan() -> str:
    """Execute the approved modification plan. This will make the actual file changes."""
    if not os.path.exists("modification_plan.json"):
        return "No modification plan found. Create one first using create_modification_plan."
    
    with open("modification_plan.json", "r") as f:
        plan = json.load(f)
    
    if plan.get("status") != "approved":
        return "Plan is not approved. User must approve the plan first."
    
    results = []
    for file_info in plan.get("files_to_modify", []):
        file_path = file_info.get("file")
        changes = file_info.get("changes")
        results.append(f"Modified {file_path}: {changes}")
    
    return "\n".join(results) if results else "No modifications executed."

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

STEP 3 - ESTIMATE SCOPE:
- How many files will I likely need to examine?
- Can I answer with search results alone, or do I need to read files?
- For large files, plan to use line ranges

STEP 4 - EXECUTE EFFICIENTLY:
- Use the minimum number of tools necessary
- Read only relevant files/sections
- Limit file reads to 3-5 files maximum unless critical

Available Tools:
- analyze_ansible_structure: Repository overview
- find_relevant_files: Find relevant files by keywords
- get_file_summary: Preview file without full read
- list_files: List directory contents
- grep_search: Regex search (returns matching lines only)
- search_in_files: Simple text search (returns matching lines only)
- read_file: Read full file or line range
- create_modification_plan: Plan code changes
- write_file: Modify a file
- execute_modification_plan: Execute modifications

Think step by step and create a clear plan before using tools."""

# System prompt for tool execution
SYSTEM_PROMPT = """You are executing a plan to help with Ansible code.

Execute the planned steps efficiently:
- Use tools as planned
- Minimize file reads
- Focus on relevant information
- Provide clear, concise responses

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
]

# Initialize the Chain of Thought agent
def create_ansible_agent():
    """Create and return the Ansible Chain of Thought agent."""
    # Use lower temperature for more focused, context-based responses
    llm = VegasChatLLM(
        usecase_name=os.getenv("VEGAS_USECASE_NAME", "AnsibleCodingAgent"),
        context_name=os.getenv("VEGAS_CONTEXT_NAME", "AnsibleCodeContext"),
        temperature=0.2,  # Lower temperature for consistent, factual responses
    )
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Node 1: Planning with Chain of Thought
    def plan_step(state: AgentState):
        """First step: Create a detailed plan with numbered steps."""
        messages = state["messages"]
        user_query = messages[-1].content if messages else ""
        
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
            return {"messages": [AIMessage(content="[All steps completed]")]}
        
        current_step_text = plan_steps[current_step_idx]
        original_query = state["messages"][0].content
        
        # Create concise execution prompt for this specific step
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
            # Execute tool calls (limit to 3 per step)
            for tool_call in response.tool_calls[:3]:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                for tool in TOOLS:
                    if tool.name == tool_name:
                        try:
                            result = tool.invoke(tool_args)
                            tool_messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"]
                            ))
                        except Exception as e:
                            tool_messages.append(ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call["id"]
                            ))
                        break
        
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
        
        # Create a summary of what was done
        summary = "\n".join([f"Step {r['step']}: {r['action']}" for r in step_results])
        
        # Ask LLM to synthesize the answer concisely
        final_prompt = HumanMessage(content=f"""Based on the plan execution, provide a CONCISE answer to the user's query.

Original Query: {original_query}

Steps Executed:
{summary}

Provide a clear, brief answer focusing only on what the user asked.""")
        
        response = llm.invoke([final_prompt])
        
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
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
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
            
            # Display the plan (if verbose mode)
            # print(f"\n[Plan]: {result['plan']}\n")
            
            # Get the final message
            final_message = result["messages"][-1]
            
            if hasattr(final_message, 'content'):
                print(f"\nAgent: {final_message.content}\n")
            else:
                print(f"\nAgent: {str(final_message)}\n")
                
        except Exception as e:
            print(f"Error: {str(e)}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()