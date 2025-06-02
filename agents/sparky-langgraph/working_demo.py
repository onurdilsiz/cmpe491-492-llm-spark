import os
import uuid
import json
import ast # For safely evaluating the string list from orchestrator if not using JSON
import time
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from functools import partial
from langchain.chat_models import init_chat_model
import PyPDF2 # Import the library
import glob   # To find files easily
# Assuming LangChain and Vertex AI libraries are installed
# pip install langchain langchain-google-vertexai google-cloud-aiplatform langchain-core langgraph
from langchain_core.messages import HumanMessage, SystemMessage
# Replace with your actual LLM initialization if different
# from langchain_google_vertexai import ChatVertexAI # Example import
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting

from langgraph.graph import StateGraph, END, START

def load_pdf_texts_from_folder(folder_path: str) -> Dict[str, str]:
    """
    Loads text from PDF files in a specified folder.

    Args:
        folder_path: The path to the folder containing the PDF files.

    Returns:
        A dictionary where keys are derived from filenames (e.g., 'jobs', 'stages')
        and values are the extracted text content from the corresponding PDFs.
    """
    pdf_texts = {}
    # Define the mapping from expected filenames to dictionary keys
    # Adjust these filenames if yours are slightly different
    filename_to_key = {
        "environment.pdf": "environment",
        "executors.pdf": "executors",
        "jobs.pdf": "jobs",
        "sql.pdf": "sql",
        "stages.pdf": "stages",
        "storage.pdf": "storage",
        # Add mappings for any other relevant PDFs you might have
    }

    print(f"Looking for PDF files in folder: {folder_path}")

    # Use glob to find all PDF files in the folder
    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        filename = os.path.basename(file_path)

        if filename in filename_to_key:
            key = filename_to_key[filename]
            print(f"Processing '{filename}' for key '{key}'...")
            try:
                with open(file_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(reader.pages)
                    text_content = ""
                    for page_num in range(num_pages):
                        page = reader.pages[page_num]
                        # Attempt to extract text, handle potential issues
                        try:
                            page_text = page.extract_text()
                            if page_text: # Add text only if extraction was successful
                                text_content += page_text + "\n\n" # Add newline between pages
                        except Exception as page_ex:
                             print(f"  Warning: Could not extract text from page {page_num + 1} of {filename}: {page_ex}")

                    if text_content:
                        pdf_texts[key] = text_content.strip()
                        print(f"  Successfully extracted text for '{key}'. Length: {len(text_content)}")
                    else:
                        print(f"  Warning: No text could be extracted from {filename}.")
                        pdf_texts[key] = f"Warning: No text could be extracted from {filename}." # Placeholder

            except FileNotFoundError:
                print(f"  Error: File not found: {file_path}")
            except PyPDF2.errors.PdfReadError as pdf_err:
                 print(f"  Error: Could not read PDF file {filename}: {pdf_err}. It might be corrupted or encrypted.")
                 pdf_texts[key] = f"Error: Could not read PDF file {filename}: {pdf_err}" # Placeholder
            except Exception as e:
                print(f"  Error processing file {filename}: {e}")
                pdf_texts[key] = f"Error processing file {filename}: {e}" # Placeholder
        else:
            print(f"Skipping file '{filename}' as it's not in the expected list.")

    if not pdf_texts:
         print("Warning: No PDF texts were loaded. Check the folder path and filenames.")

    return pdf_texts

llm = init_chat_model("google_vertexai:gemini-2.5-pro-preview-03-25")



# --- 1. State Definition (MODIFIED) ---

# Define the merge function for specialist_findings
def merge_findings(existing_findings: Dict[str, str], new_finding: Dict[str, str]) -> Dict[str, str]:
    """Merges new findings into the existing findings dictionary."""
    # Ensure inputs are dictionaries
    if existing_findings is None:
        existing_findings = {}
    if new_finding is None:
        new_finding = {}
    # Create a copy to avoid modifying the original state directly in unexpected ways
    updated_findings = existing_findings.copy()
    updated_findings.update(new_finding)
    return updated_findings

# Add metrics merge function
def merge_metrics(existing_metrics: Dict[str, Any], new_metric: Dict[str, Any]) -> Dict[str, Any]:
    """Merges new metrics into the existing metrics dictionary."""
    if not isinstance(existing_metrics, dict):
        existing_metrics = {"total_tokens": 0, "total_latency": 0.0, "llm_calls": []}
    
    if not isinstance(new_metric, dict):
        new_metric = {"total_tokens": 0, "total_latency": 0.0, "llm_calls": []}
    
    if "total_tokens" not in existing_metrics:
        existing_metrics["total_tokens"] = 0
    if "total_latency" not in existing_metrics:
        existing_metrics["total_latency"] = 0.0
    if "llm_calls" not in existing_metrics:
        existing_metrics["llm_calls"] = []
    
    updated_metrics = existing_metrics.copy()
    updated_metrics["total_tokens"] += new_metric.get("total_tokens", 0)
    updated_metrics["total_latency"] += new_metric.get("total_latency", 0.0)
    updated_metrics["llm_calls"].extend(new_metric.get("llm_calls", []))
    
    return updated_metrics

# Use BaseState for more explicit state management features if desired, or stick with TypedDict
class AgentState(TypedDict):
    session_id: str
    user_query: str
    pdf_texts: Dict[str, str]
    conversation_history: List[Dict[str, str]]

    # Intermediate fields
    agents_to_call: List[str]
    # *** MODIFICATION HERE ***
    # Use Annotated to specify the merge function for concurrent updates
    specialist_findings: Annotated[Dict[str, str], merge_findings]
    
    # Metrics tracking
    metrics: Annotated[Dict[str, Any], merge_metrics]

    # Final output fields
    final_analysis: Optional[str]
    final_response: Optional[str]

# --- 2. Utility Functions (Keep as before) ---
def format_response(analysis: str) -> str:
    return f"Based on the provided Spark UI PDF data:\n\n{analysis}"

def format_history_for_prompt(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    recent_history = history[-max_turns * 2:]
    formatted = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent_history])
    return f"Relevant recent conversation history:\n{formatted}" if formatted else "No previous conversation history."

# --- 3. Agent Node Functions (Keep specialist agent return format) ---

# Orchestrator (keep as before)
def track_llm_call(agent_name: str, llm: BaseChatModel, messages: List, description: str = "") -> tuple[str, Dict[str, Any]]:
    """
    Wrapper function to track LLM calls with latency and token usage.
    Returns: (response_content, metrics_dict)
    """
    start_time = time.time()
    
    try:
        response = llm.invoke(messages)
        end_time = time.time()
        latency = end_time - start_time
        
        # Extract token usage from response if available
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        # Try to get token usage from response metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('usage', {})
            if usage:
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0) 
                total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
        
        # If no usage metadata, estimate tokens (rough approximation: 1 token ≈ 4 characters)
        if total_tokens == 0:
            input_text = " ".join([msg.content for msg in messages])
            input_tokens = len(input_text) // 4
            output_tokens = len(response.content) // 4
            total_tokens = input_tokens + output_tokens
        
        metrics = {
            "total_tokens": total_tokens,
            "total_latency": latency,
            "llm_calls": [{
                "agent": agent_name,
                "description": description,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "timestamp": time.time()
            }]
        }
        
        print(f"[METRICS] {agent_name}: {latency:.2f}s, {total_tokens} tokens ({input_tokens} in, {output_tokens} out)")
        
        return response.content, metrics
        
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        
        metrics = {
            "total_tokens": 0,
            "total_latency": latency,
            "llm_calls": [{
                "agent": agent_name,
                "description": f"{description} (ERROR)",
                "latency": latency,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error": str(e),
                "timestamp": time.time()
            }]
        }
        
        raise e

# Orchestrator (keep as before)
def orchestrator_router(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    print("--- Running Orchestrator ---")
    query = state["user_query"]
    history = state["conversation_history"]
    formatted_history = format_history_for_prompt(history)
    available_agents = ['jobs', 'stages', 'executors', 'sql', 'storage', 'environment']

    prompt_messages = [
        SystemMessage(content=f"""
You are an expert system orchestrator for Spark performance analysis.
Your task is to determine which Spark UI tabs (and thus corresponding specialist agents) are most relevant to answering the current user query, considering the conversation history.
Available specialist agents correspond to these Spark UI tabs: {available_agents}.
Respond ONLY with a JSON list of agent names (strings) based on the tab keys. For example: ["jobs", "stages"].
If the query is broad or unclear, include the most common agents: ["jobs", "stages", "executors"].
If no specific tabs seem relevant, return an empty list [].
"""),
        HumanMessage(content=f"""
Conversation History:
{formatted_history}

Current User Query: "{query}"

Based on the query and history, which specialist agents are needed? Respond only with the JSON list.
""")
    ]
    
    try:
        content, metrics = track_llm_call("orchestrator", llm, prompt_messages, "Selecting relevant agents")
        
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
        content = content.strip()
        agents_to_call_keys = json.loads(content)
        if not isinstance(agents_to_call_keys, list) or not all(isinstance(item, str) for item in agents_to_call_keys):
             raise ValueError("LLM did not return a valid list of strings.")
        agents_to_call_keys = [agent for agent in agents_to_call_keys if agent in available_agents]
    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Orchestrator LLM call or parsing failed: {e}. Falling back to default agents.")
        agents_to_call_keys = ["jobs", "stages", "executors"]
        # Create error metrics
        metrics = {
            "total_tokens": 0,
            "total_latency": 0.0,
            "llm_calls": [{
                "agent": "orchestrator",
                "description": "Selecting relevant agents (FALLBACK)",
                "latency": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error": str(e),
                "timestamp": time.time()
            }]
        }

    print(f"Orchestrator decided to call agents for tabs: {agents_to_call_keys}")
    specialist_agent_names = [f"{key}_agent" for key in agents_to_call_keys]

    return {"agents_to_call": specialist_agent_names, "specialist_findings": {}, "metrics": metrics}


# Specialist Agent Factory (Ensure it returns dict like {agent_node_name: findings})
def create_specialist_node(agent_name: str, tab_key: str, llm: BaseChatModel):
    def agent_node(state: AgentState) -> Dict[str, Any]:
        print(f"--- Running Specialist: {agent_name} (Tab: {tab_key}) ---")
        
        query = state["user_query"]
        history = state["conversation_history"]
        formatted_history = format_history_for_prompt(history)
        pdf_text = state["pdf_texts"].get(tab_key, f"PDF text not found for the '{tab_key}' tab.")
        max_text_length = 8000
        truncated_text = pdf_text[:max_text_length]
        if len(pdf_text) > max_text_length: truncated_text += "\n... [Text truncated]"

        # Specialized prompts for each tab
        tab_specific_prompts = {
            "executors": """You are a Spark performance expert focused on executor analysis. Analyze the following text from the 'executors' tab.
Key areas to examine:
1. Autoscaling behavior - Check for sharp increases in executor count after start
2. Memory usage and potential OOM issues:
   - Look for exit codes 143 or 137 indicating executor kills
   - Check if executors are using available memory efficiently
   - Identify memory spills to disk
3. Resource utilization patterns
4. Executor lifetime and stability

Provide specific metrics when available. If you spot potential issues, suggest concrete optimization parameters.""",

            "stages": """You are a Spark performance expert focused on stage analysis. Analyze the following text from the 'stages' tab.
Key areas to examine:
1. Data skew detection:
   - Compare min/median/max partition sizes
   - Flag cases where max size is >10x median
2. Shuffle partition sizes and counts
3. Stage duration patterns
4. Task distribution and parallelism
5. Identify bottleneck stages

Focus on concrete metrics and flag any significant imbalances or inefficiencies.""",

            "environment": """You are a Spark performance expert focused on Spark configuration analysis. Analyze the following text from the 'environment' tab.
Key areas to examine:
1. Auto Broadcast Join Threshold:
   - Check if it's enabled (should not be -1)
   - Suggest adjustments based on data sizes
2. Adaptive Query Execution (AQE) settings
3. Memory-related configurations:
   - Executor memory settings
   - Memory overhead factors
4. Dynamic allocation settings:
   - Initial/min/max executors
   - Allocation ratio
   - Queue backlog settings

Provide specific configuration recommendations when issues are identified.""",

            "jobs": """You are a Spark performance expert focused on job analysis. Analyze the following text from the 'jobs' tab.
Key areas to examine:
1. Identify top 5 longest-running jobs
2. Analyze job execution patterns
3. Job dependencies and parallelism
4. Success/failure patterns
5. Resource utilization across jobs

Focus on concrete metrics and timing data to support your analysis.""",

            "sql": """You are a Spark performance expert focused on SQL query analysis. Analyze the following text from the 'sql' tab.
Key areas to examine:
1. Shuffle partition configurations
2. Join strategy selection
3. Query plan optimization
4. AQE behavior and impacts
5. Partition sizes and counts

Look for opportunities to optimize query execution through configuration adjustments.""",

            "storage": """You are a Spark performance expert focused on storage analysis. Analyze the following text from the 'storage' tab.
Key areas to examine:
1. Cache utilization and efficiency
2. Storage level choices
3. Memory vs. disk storage patterns
4. Partition distribution
5. Data persistence strategies

Identify potential storage-related bottlenecks and optimization opportunities."""
        }

        system_prompt = tab_specific_prompts.get(
            tab_key,
            f"You are a Spark performance analysis expert focused on the '{tab_key}' tab of the Spark UI."
        )

        prompt_messages = [
            SystemMessage(content=f"""{system_prompt}
Analyze the provided text in the context of the user's query and conversation history.
Focus only on what can be inferred from the provided text. Do not invent data.
If the text is irrelevant to the query, state that clearly.
Provide a concise summary of your findings with specific metrics when available."""),
            HumanMessage(content=f"""
Conversation History:
{formatted_history}

User Query: "{query}"

'{tab_key}' Tab Text:
--- START TEXT ---
{truncated_text}
--- END TEXT ---

Based only on the text above and the query context, what are your findings?
""")
        ]

        try:
            content, metrics = track_llm_call(agent_name, llm, prompt_messages, f"Analyzing {tab_key} tab")
            findings = {agent_name: content}
        except Exception as e:
            print(f"Error in {agent_name} LLM call: {e}")
            findings = {agent_name: f"Error analyzing '{tab_key}' tab: {e}"}
            metrics = {
                "total_tokens": 0,
                "total_latency": 0.0,
                "llm_calls": [{
                    "agent": agent_name,
                    "description": f"Analyzing {tab_key} tab (ERROR)",
                    "latency": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "error": str(e),
                    "timestamp": time.time()
                }]
            }

        return {"specialist_findings": findings, "metrics": metrics}

    return agent_node

# Synthesizer (Keep as before)
def synthesizer_agent(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    print("--- Running Synthesizer ---")
    query = state["user_query"]
    history = state["conversation_history"]
    formatted_history = format_history_for_prompt(history, max_turns=8)
    findings = state["specialist_findings"] # This will now be the merged dictionary

    if not findings:
        analysis = "I could not find specific information relevant to your query in the provided PDF sections that were analyzed. Could you please specify which aspects of the Spark job you are interested in?"
        metrics = {
            "total_tokens": 0,
            "total_latency": 0.0,
            "llm_calls": []
        }
        return {"final_analysis": analysis, "metrics": metrics}

    findings_str = "\n\n".join([f"--- Findings from {agent_name.replace('_agent', '')} Tab ---:\n{result}"
                                for agent_name, result in findings.items()])

    prompt_messages = [
         SystemMessage(content="""
You are a Spark performance optimization expert consolidating analysis from different Spark UI tab specialists.
Your task is to synthesize the provided findings into a coherent analysis addressing the user's query, considering the conversation history.
Identify potential root causes by correlating information across different findings.
Provide clear, actionable optimization suggestions based *only* on the evidence presented in the findings and general Spark best practices.
If findings are contradictory or insufficient, acknowledge that.
Structure your response clearly: first the overall analysis, then specific suggestions.
Ensure your response flows logically from the conversation history.
"""),
         HumanMessage(content=f"""
Conversation History:
{formatted_history}

Original User Query for this turn: "{query}"

Collected Findings from Specialist Agents:
{findings_str}

Based on the query, history, and these findings, please provide a synthesized analysis and actionable optimization suggestions.
""")
     ]
    try:
        content, metrics = track_llm_call("synthesizer", llm, prompt_messages, "Synthesizing findings")
        analysis = content
    except Exception as e:
        print(f"Error in Synthesizer LLM call: {e}")
        analysis = f"Error synthesizing the analysis: {e}"
        metrics = {
            "total_tokens": 0,
            "total_latency": 0.0,
            "llm_calls": [{
                "agent": "synthesizer",
                "description": "Synthesizing findings (ERROR)",
                "latency": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error": str(e),
                "timestamp": time.time()
            }]
        }

    return {"final_analysis": analysis, "metrics": metrics}

# Formatter (Keep as before)
def response_formatter_node(state: AgentState) -> Dict[str, Any]:
    print("--- Running Formatter ---")
    analysis = state["final_analysis"]
    if not analysis:
        if not state.get("specialist_findings"):
             final_response = "I reviewed the relevant PDF sections based on your query, but couldn't find specific details to generate an analysis. You might need to check other tabs or provide more context."
        else:
             final_response = "I encountered an issue while synthesizing the findings. Please try rephrasing your query."
    else:
        final_response = format_response(analysis)
    return {"final_response": final_response}


# --- 4. Graph Definition and Conditional Routing (Keep as before) ---

workflow = StateGraph(AgentState)

# Add nodes (no change here, partial still works)
workflow.add_node("orchestrator", partial(orchestrator_router, llm=llm))
pdf_tab_keys = ["jobs", "stages", "executors", "sql", "storage", "environment"]
specialist_node_names = {}
for key in pdf_tab_keys:
    node_name = f"{key}_agent"
    specialist_node_names[key] = node_name
    workflow.add_node(node_name, create_specialist_node(node_name, key, llm))

workflow.add_node("synthesizer", partial(synthesizer_agent, llm=llm))
workflow.add_node("formatter", response_formatter_node)

# Define Edges (no change here)
workflow.set_entry_point("orchestrator")

def route_to_specialists(state: AgentState):
    agent_node_names_to_call = state.get("agents_to_call", [])
    if not agent_node_names_to_call:
        print("Routing: Orchestrator -> Synthesizer (no specialists needed)")
        return "synthesizer"
    else:
        print(f"Routing: Orchestrator -> Specialists ({agent_node_names_to_call})")
        return agent_node_names_to_call

workflow.add_conditional_edges(
    "orchestrator",
    route_to_specialists,
    {
        "synthesizer": "synthesizer",
        **{node_name: node_name for node_name in specialist_node_names.values()}
    }
)

for node_name in specialist_node_names.values():
    workflow.add_edge(node_name, "synthesizer")

workflow.add_edge("synthesizer", "formatter")
workflow.add_edge("formatter", END)

# Compile the graph (no change here)
print("Compiling LangGraph application...")
try:
    app = workflow.compile()
    print("Graph compiled successfully.")
except Exception as e:
    print(f"Error compiling graph: {e}")
    exit()


# --- 5. Simulation of Session Manager and Chat Loop (Keep as before) ---
session_storage = {}

def display_metrics_summary(metrics: Dict[str, Any]):
    """Display a summary of token usage and latency metrics."""
    print("\n" + "="*50)
    print("📊 PERFORMANCE METRICS SUMMARY")
    print("="*50)
    
    total_tokens = metrics.get("total_tokens", 0) if isinstance(metrics, dict) else 0
    total_latency = metrics.get("total_latency", 0.0) if isinstance(metrics, dict) else 0.0
    llm_calls = metrics.get("llm_calls", []) if isinstance(metrics, dict) else []
    overall_time = metrics.get("overall_execution_time", 0.0) if isinstance(metrics, dict) else 0.0
    
    print(f"🔢 Total Tokens Used: {total_tokens:,}")
    print(f"⏱️  Total LLM Latency: {total_latency:.2f} seconds")
    print(f"🕐 Overall Execution Time: {overall_time:.2f} seconds")
    print(f"📞 Number of LLM Calls: {len(llm_calls)}")
    
    if total_tokens > 0 and llm_calls:
        avg_tokens_per_call = total_tokens / len(llm_calls)
        print(f"📈 Average Tokens per Call: {avg_tokens_per_call:.1f}")
    
    if total_latency > 0 and llm_calls:
        avg_latency_per_call = total_latency / len(llm_calls)
        print(f"⚡ Average Latency per Call: {avg_latency_per_call:.2f}s")
    
    if llm_calls:
        print("\n📋 Detailed Call Breakdown:")
        print("-" * 50)
        
        for i, call in enumerate(llm_calls, 1):
            agent = call.get("agent", "unknown")
            description = call.get("description", "")
            latency = call.get("latency", 0.0)
            tokens = call.get("total_tokens", 0)
            input_tokens = call.get("input_tokens", 0)
            output_tokens = call.get("output_tokens", 0)
            error = call.get("error")
            
            status = "❌ ERROR" if error else "✅ SUCCESS"
            print(f"{i:2d}. {agent:20} | {status:10} | {latency:6.2f}s | {tokens:5d} tokens ({input_tokens}→{output_tokens})")
            if description:
                print(f"    📝 {description}")
            if error:
                print(f"    ⚠️  Error: {error}")
    else:
        print("\n📋 No LLM calls were tracked.")
    
    if "error" in metrics:
        print(f"\n⚠️  Session Error: {metrics['error']}")
    
    print("="*50)

def run_chat_turn(session_id: str, user_query: str, pdf_texts: Optional[Dict[str, str]] = None):
    global session_storage
    
    # Track overall execution time
    overall_start_time = time.time()
    
    if session_id not in session_storage:
        if pdf_texts is None: 
            return "Error: PDF texts must be provided for the first turn of a session."
        print(f"Initializing new session: {session_id}")
        session_storage[session_id] = { "conversation_history": [], "pdf_texts": pdf_texts }
    elif pdf_texts is not None:
        print(f"Warning: PDF texts provided for existing session {session_id}. Overwriting.")
        session_storage[session_id]["pdf_texts"] = pdf_texts

    current_session = session_storage[session_id]
    current_history = current_session["conversation_history"]
    current_pdfs = current_session["pdf_texts"]

    # *** IMPORTANT: Initialize specialist_findings in the input state ***
    # The merge function needs an initial value to merge into.
    graph_input: AgentState = {
        "session_id": session_id,
        "user_query": user_query,
        "pdf_texts": current_pdfs,
        "conversation_history": current_history.copy(),
        "agents_to_call": [],
        "specialist_findings": {}, # Initialize as empty dict
        "metrics": {"total_tokens": 0, "total_latency": 0.0, "llm_calls": []},
        "final_analysis": None,
        "final_response": None,
    }

    print(f"\n--- Invoking Graph for Session {session_id} ---")
    try:
        final_state = app.invoke(graph_input)
        agent_response = final_state.get("final_response", "Sorry, I encountered an issue processing your request.")
        
        overall_end_time = time.time()
        overall_execution_time = overall_end_time - overall_start_time
        
        print("--- Graph Invocation Complete ---")
        
        # Display metrics
        metrics = final_state.get("metrics", {})
        metrics["overall_execution_time"] = overall_execution_time
        display_metrics_summary(metrics)
        
    except Exception as e:
        overall_end_time = time.time()
        overall_execution_time = overall_end_time - overall_start_time
        
        print(f"--- Graph Invocation ERROR: {e} ---")
        agent_response = f"Sorry, an error occurred during processing: {e}. Please check the logs."
        
        # Display error metrics
        error_metrics = {
            "total_tokens": 0,
            "total_latency": 0.0,
            "llm_calls": [],
            "overall_execution_time": overall_execution_time,
            "error": str(e)
        }
        display_metrics_summary(error_metrics)

    current_session["conversation_history"].append({"role": "user", "content": user_query})
    current_session["conversation_history"].append({"role": "assistant", "content": agent_response})
    return agent_response

      
# --- Example Usage (Interactive Chat Loop) ---

if __name__ == "__main__":
    # --- Initialization ---
    
    pdf_folder = "runs/5 - Memory Spill Demo" 
    print(f"Attempting to load PDFs from: {os.path.abspath(pdf_folder)}")

    try:
        # Load PDF texts from the specified folder
        actual_pdf_texts = load_pdf_texts_from_folder(pdf_folder)
    except Exception as load_err:
        print(f"FATAL ERROR: Could not load PDF files: {load_err}")
        actual_pdf_texts = {}

    # Check if any PDFs were loaded successfully
    if not actual_pdf_texts:
        print("\nERROR: No PDF data was loaded. Cannot start chat session.")
        print("Please ensure the PDF files exist in the specified folder and are readable.")
        print("Expected filenames: environment.pdf, executors.pdf, jobs.pdf, sql.pdf, stages.pdf, storage.pdf")
        exit() # Exit if no PDFs are loaded
    else:
        print("\nPDF Loading Summary:")
        for key, text in actual_pdf_texts.items():
            print(f"  - Loaded '{key}': {len(text)} characters")
        print("-" * 30)

    # --- Initialize Session ---
    my_session_id = str(uuid.uuid4())
    print(f"Initializing new session: {my_session_id}")
    # Store initial state in session_storage
    session_storage[my_session_id] = {
        "conversation_history": [],
        "pdf_texts": actual_pdf_texts # Store PDF texts for the session
    }

    print("\nChat session started.")
    print("Enter your query about the Spark application based on the loaded PDFs.")
    print("Type 'quit' or 'exit' to end the session.")
    print("-" * 30)

    # --- Chat Loop ---
    while True:
        try:
            user_query = input("You: ")
        except EOFError:
            # Handle Ctrl+D or similar EOF signals gracefully
            print("\nExiting...")
            break

        if user_query.lower().strip() in ["quit", "exit"]:
            print("Assistant: Goodbye!")
            break

        if not user_query.strip():
            # Handle empty input if desired, or just loop again
            continue

        # Call the chat turn function. It will use the existing session ID
        # and retrieve history/PDFs from session_storage.
        # We don't pass pdf_texts here after initialization.
        assistant_response = run_chat_turn(my_session_id, user_query)

        print(f"Assistant: {assistant_response}")
        print("-" * 10) # Separator for clarity

    print("\nChat session ended.")

    # Optional: Clean up session data if needed
    # del session_storage[my_session_id]

    