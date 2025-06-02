import os
import uuid
import json
import ast
import time
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from functools import partial
from langchain.chat_models import init_chat_model
import PyPDF2
import glob
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END, START

def load_pdf_texts_from_folder(folder_path: str) -> Dict[str, str]:
    """
    Loads text from PDF files in a specified folder.
    """
    pdf_texts = {}
    filename_to_key = {
        "environment.pdf": "environment",
        "executors.pdf": "executors",
        "jobs.pdf": "jobs",
        "sql.pdf": "sql",
        "stages.pdf": "stages",
        "storage.pdf": "storage",
    }

    print(f"Looking for PDF files in folder: {folder_path}")

    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        filename = os.path.basename(file_path)
        if filename in filename_to_key:
            key = filename_to_key[filename]
            print(f"Processing '{filename}' for key '{key}'...")
            try:
                with open(file_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text_content = ""
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text_content += page_text + "\n\n"
                        except Exception as page_ex:
                            print(f"  Warning: Could not extract text from page {page_num + 1} of {filename}: {page_ex}")

                    if text_content:
                        pdf_texts[key] = text_content.strip()
                        print(f"  Successfully extracted text for '{key}'. Length: {len(text_content)}")
                    else:
                        print(f"  Warning: No text could be extracted from {filename}.")
                        pdf_texts[key] = f"Warning: No text could be extracted from {filename}."

            except Exception as e:
                print(f"  Error processing file {filename}: {e}")
                pdf_texts[key] = f"Error processing file {filename}: {e}"
        else:
            print(f"Skipping file '{filename}' as it's not in the expected list.")

    if not pdf_texts:
        print("Warning: No PDF texts were loaded. Check the folder path and filenames.")

    return pdf_texts

llm = init_chat_model("openai:gpt-4o") # google_vertexai:gemini-2.0-flash-001

# Define the merge function for case findings
def merge_case_findings(existing_findings: Dict[str, str], new_finding: Dict[str, str]) -> Dict[str, str]:
    """Merges new case findings into the existing findings dictionary."""
    if existing_findings is None:
        existing_findings = {}
    if new_finding is None:
        new_finding = {}
    updated_findings = existing_findings.copy()
    updated_findings.update(new_finding)
    return updated_findings

# Add metrics merge function
def merge_metrics(existing_metrics: Dict[str, Any], new_metric: Dict[str, Any]) -> Dict[str, Any]:
    """Merges new metrics into the existing metrics dictionary."""
    # Ensure we have proper dictionaries with default structure
    if not isinstance(existing_metrics, dict):
        existing_metrics = {"total_tokens": 0, "total_latency": 0.0, "llm_calls": []}
    
    if not isinstance(new_metric, dict):
        new_metric = {"total_tokens": 0, "total_latency": 0.0, "llm_calls": []}
    
    # Initialize existing_metrics with defaults if keys are missing
    if "total_tokens" not in existing_metrics:
        existing_metrics["total_tokens"] = 0
    if "total_latency" not in existing_metrics:
        existing_metrics["total_latency"] = 0.0
    if "llm_calls" not in existing_metrics:
        existing_metrics["llm_calls"] = []
    
    # Create updated metrics
    updated_metrics = existing_metrics.copy()
    
    # Safely merge new metrics
    updated_metrics["total_tokens"] += new_metric.get("total_tokens", 0)
    updated_metrics["total_latency"] += new_metric.get("total_latency", 0.0)
    updated_metrics["llm_calls"].extend(new_metric.get("llm_calls", []))
    
    return updated_metrics

class CaseAgentState(TypedDict):
    session_id: str
    user_query: str
    pdf_texts: Dict[str, str]
    conversation_history: List[Dict[str, str]]
    
    # Intermediate fields
    cases_to_check: List[str]
    case_findings: Annotated[Dict[str, str], merge_case_findings]
    
    # Metrics tracking
    metrics: Annotated[Dict[str, Any], merge_metrics]
    
    # Final output fields
    final_analysis: Optional[str]
    final_response: Optional[str]

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

def format_response(analysis: str) -> str:
    return f"Based on the provided Spark UI PDF data:\n\n{analysis}"

def format_history_for_prompt(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    recent_history = history[-max_turns * 2:]
    formatted = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent_history])
    return f"Relevant recent conversation history:\n{formatted}" if formatted else "No previous conversation history."

# Modified Case Orchestrator with better error handling
def case_orchestrator(state: CaseAgentState, llm: BaseChatModel) -> Dict[str, Any]:
    print("--- Running Case Orchestrator ---")
    query = state["user_query"]
    history = state["conversation_history"]
    formatted_history = format_history_for_prompt(history)
    
    available_cases = [
        'skewed_join', 'huge_collect', 'too_many_partitions', 'cache_no_unpersist',
      'cartesian_join', 'many_small_files', 
         'rdd_conversion', 'autoscaling_backlog', 'broadcast_threshold',
        'memory_spill', 'gc_heavy_rdd'
    ]

    prompt_messages = [
        SystemMessage(content=f"""
You are an expert Spark performance issue detector and orchestrator.
Your task is to determine which specific performance issue cases are most relevant to investigate based on the user's query and conversation history.

Available performance issue cases to investigate: {available_cases}

Case descriptions:
- skewed_join: Skewed shuffle with most data on 1 key, broadcast disabled, single-file writes
- huge_collect: Large collect() operations causing driver memory spikes
- too_many_partitions: Excessive shuffle partitions (like 10000) causing overhead
- cache_no_unpersist: Large DataFrames cached but never released
- cartesian_join: Cartesian joins creating massive data explosion
- many_small_files: Writing thousands of tiny files
- multi_cache: Multiple large caches hogging executor memory
- autoscaling_backlog: Dynamic allocation starting with too few executors
- broadcast_threshold: Broadcast joins blocked by threshold limits
- memory_spill: Execution memory cuts causing shuffle spill to disk
- gc_heavy_rdd: RDD operations creating billions of tiny objects causing heavy GC

Respond ONLY with a JSON list of case names (strings). For example: ["skewed_join", "memory_spill"].
If the query is general, include the most common cases: ["skewed_join", "memory_spill", "autoscaling_backlog"].
If the query is about all cases, include all cases: ["skewed_join", "memory_spill", "autoscaling_backlog", "huge_collect", "too_many_partitions", "cartesian_join", "many_small_files","autoscaling_backlog", "broadcast_threshold", "memory_spill",  "gc_heavy_rdd"].
If no specific cases seem relevant, return an empty list [].
"""),
        HumanMessage(content=f"""
Conversation History:
{formatted_history}

Current User Query: "{query}"

Based on the query and history, which performance issue cases should be investigated? Respond only with the JSON list.
""")
    ]
    
    try:
        content, metrics = track_llm_call("case_orchestrator", llm, prompt_messages, "Selecting relevant cases")
        
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
        content = content.strip()
        cases_to_check = json.loads(content)
        if not isinstance(cases_to_check, list) or not all(isinstance(item, str) for item in cases_to_check):
            raise ValueError("LLM did not return a valid list of strings.")
        cases_to_check = [case for case in cases_to_check if case in available_cases]
    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Case Orchestrator LLM call or parsing failed: {e}. Falling back to default cases.")
        cases_to_check = ["skewed_join", "memory_spill", "autoscaling_backlog"]
        # Create error metrics
        metrics = {
            "total_tokens": 0,
            "total_latency": 0.0,
            "llm_calls": [{
                "agent": "case_orchestrator",
                "description": "Selecting relevant cases (FALLBACK)",
                "latency": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error": str(e),
                "timestamp": time.time()
            }]
        }

    print(f"Case Orchestrator decided to check cases: {cases_to_check}")
    case_agent_names = [f"{case}_agent" for case in cases_to_check]

    return {"cases_to_check": case_agent_names, "case_findings": {}, "metrics": metrics}

# Modified Case Agent Factory with tracking
def create_case_agent(case_name: str, llm: BaseChatModel):
    def case_agent(state: CaseAgentState) -> Dict[str, Any]:
        agent_name = f"{case_name}_agent"
        print(f"--- Running Case Agent: {agent_name} ---")
        
        query = state["user_query"]
        history = state["conversation_history"]
        formatted_history = format_history_for_prompt(history)
        pdf_texts = state["pdf_texts"]
        
        # Combine all PDF texts for comprehensive analysis
        all_texts = "\n\n".join([f"=== {tab.upper()} TAB ===\n{text}" for tab, text in pdf_texts.items()])
        max_text_length = 15000  # Increased for comprehensive analysis
        truncated_text = all_texts[:max_text_length]
        if len(all_texts) > max_text_length:
            truncated_text += "\n... [Text truncated]"

        # Case-specific detection prompts
        case_prompts = {
            "skewed_join": """
You are a Spark performance expert detecting SKEWED JOIN issues.

Look for these specific symptoms:
1. SQL/DAG tab: Red "⚠ Skewed" badge or mentions of skew
2. One straggler task lasting orders-of-magnitude longer than others
3. Storage: Only one output Parquet part-file (indicating coalesce(1))
4. Broadcast joins disabled when they should be enabled
5. Highly imbalanced partition sizes in shuffle operations

Detection criteria:
- Task duration variance (max >> median)
- Mentions of "skew", "imbalanced", or "straggler"
- Single output file patterns
- Broadcast join threshold issues
""",
            
            "huge_collect": """
You are a Spark performance expert detecting HUGE COLLECT issues.

Look for these specific symptoms:
1. Executors → Driver: GC Time and memory spikes
2. Stages finish but job stalls during "collect" operation
3. Driver memory usage jumping significantly
4. collect() operations mentioned in logs
5. Driver OOM errors or memory pressure

Detection criteria:
- Driver memory/GC spikes
- Job stalling after stage completion
- Mentions of "collect", "driver memory", or "OOM"
""",
            
            "too_many_partitions": """
You are a Spark performance expert detecting TOO MANY PARTITIONS issues.

Look for these specific symptoms:
1. Stages list: Two identical stages with 10,000+ tasks
2. spark.sql.shuffle.partitions set to very high values (like 10000)
3. Summary metrics: High Scheduler Delay vs Task Time
4. Duplicate shuffle operations
5. Task overhead dominating actual processing time

Detection criteria:
- Very high partition counts (>1000)
- High scheduler delay
- Many small tasks
- Duplicate shuffle stages
""",
            
            "cache_no_unpersist": """
You are a Spark performance expert detecting CACHE WITHOUT UNPERSIST issues.

Look for these specific symptoms:
1. Storage tab: 100% cached DataFrames
2. Later jobs showing rising GC times
3. Memory pressure without unpersist calls
4. Cached data persisting through entire job
5. Spill to disk due to cache memory pressure

Detection criteria:
- High cache usage percentages
- Rising GC times over job duration
- Memory spill increases
- No unpersist operations mentioned
""",
            
            "python_udf": """
You are a Spark performance expert detecting PYTHON UDF issues.

Look for these specific symptoms:
1. SQL Plan: "*> PythonUDF" operations
2. Task metrics: High CPU time spent in Python
3. No whole-stage code generation (codegen disabled)
4. Serialization overhead between JVM and Python
5. Performance degradation from vectorized operations

Detection criteria:
- PythonUDF in execution plans
- Python execution time metrics
- Disabled code generation
- Serialization overhead
""",
            
            "cartesian_join": """
You are a Spark performance expert detecting CARTESIAN JOIN issues.

Look for these specific symptoms:
1. Plan: CartesianProduct operations
2. Shuffle read data exploding (rows × 100 or more)
3. Massive increase in data size during joins
4. Cross joins without proper conditions
5. Exponential growth in processing time

Detection criteria:
- CartesianProduct in plans
- Massive data size increases
- Cross join operations
- Exponential row count growth
""",
            
            "many_small_files": """
You are a Spark performance expert detecting MANY SMALL FILES issues.

Look for these specific symptoms:
1. Job I/O phase spawning 5,000+ tasks
2. Output directory with thousands of part-files
3. File listing operations taking excessive time
4. Many small partition writes
5. GCS/HDFS metadata overhead

Detection criteria:
- Very high number of output tasks
- Many part-files mentioned
- File system overhead
- Small file sizes in output
""",
            
            "no_compression": """
You are a Spark performance expert detecting NO COMPRESSION issues.

Look for these specific symptoms:
1. Environment tab: parquet.compression=none or similar
2. Output folder sizes much larger than expected
3. I/O bottlenecks due to uncompressed data
4. Network transfer overhead
5. Storage cost inefficiencies

Detection criteria:
- Compression settings disabled
- Large output sizes
- I/O performance issues
- Uncompressed format mentions
""",
            
            "multi_cache": """
You are a Spark performance expert detecting MULTIPLE CACHE issues.

Look for these specific symptoms:
1. Storage: Multiple large cached DataFrames
2. Executors: Memory usage close to capacity
3. Rising GC times as caches accumulate
4. Memory pressure from over-caching
5. Potential cache evictions

Detection criteria:
- Multiple large cached datasets
- High memory utilization
- Increasing GC pressure
- Cache-related memory issues
""",
            
            "rdd_conversion": """
You are a Spark performance expert detecting UNNECESSARY RDD CONVERSION issues.

Look for these specific symptoms:
1. Plan: No columnar/vectorized operations
2. Extra serialization stages
3. DataFrame → RDD → DataFrame round-trips
4. Loss of Catalyst optimizations
5. Performance degradation from conversions

Detection criteria:
- RDD operations in DataFrame workflows
- Serialization overhead
- Loss of vectorization
- Unnecessary conversions mentioned
""",
            
            "autoscaling_backlog": """
You are a Spark performance expert detecting AUTOSCALING BACKLOG issues.

Look for these specific symptoms:
1. Executors tab (Timeline): Long period with minimal executors
2. Task queue/backlog > 0 for extended periods
3. Later rapid scale-up of executors
4. Initial executor count too low
5. Dynamic allocation delays

Detection criteria:
- Low initial executor counts
- Task backlogs
- Delayed executor allocation
- Autoscaling timeline issues
""",
            
            "broadcast_threshold": """
You are a Spark performance expert detecting BROADCAST THRESHOLD issues.

Look for these specific symptoms:
1. Plan: ShuffleHashJoin instead of BroadcastHashJoin
2. SQL metrics: Sizable shuffle read/write for small tables
3. Broadcast join blocked by threshold limits (like 1MB)
4. Small table joins using shuffle instead of broadcast
5. Suboptimal join strategy selection

Detection criteria:
- Shuffle joins for small tables
- Broadcast threshold configuration
- Join strategy inefficiencies
- Small table shuffle operations
""",
            
            "memory_spill": """
You are a Spark performance expert detecting MEMORY SPILL issues.

Look for these specific symptoms:
1. SQL/Stage metrics: "Memory spilled" values > 0
2. Executors: High disk I/O during processing
3. Execution memory fraction too low (like 0.1)
4. Spill to disk operations
5. Memory pressure causing disk writes

Detection criteria:
- Non-zero spill metrics
- High disk I/O
- Memory pressure indicators
- Spill-to-disk operations
""",
            
            "pandas_udf": """
You are a Spark performance expert detecting INEFFICIENT PANDAS UDF issues.

Look for these specific symptoms:
1. Plan: PandasUDF operations
2. High deserialization and Python processing time
3. Arrow batch processing underutilized
4. Row-wise operations where vectorized built-ins would work
5. Serialization overhead between JVM and Python

Detection criteria:
- PandasUDF in execution plans
- High Python processing overhead
- Inefficient vectorization
- Unnecessary Pandas operations
""",
            
            "gc_heavy_rdd": """
You are a Spark performance expert detecting GC HEAVY RDD issues.

Look for these specific symptoms:
1. Executors: GC Time bars turning red
2. Low CPU utilization despite high processing
3. Job wall-clock time much greater than CPU time
4. RDD flatMap creating billions of tiny objects
5. Heavy garbage collection pressure

Detection criteria:
- Excessive GC times
- Low CPU efficiency
- Object creation overhead
- Memory pressure from small objects
"""
        }

        system_prompt = case_prompts.get(case_name, f"You are analyzing for {case_name} performance issues.")

        prompt_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Conversation History:
{formatted_history}

User Query: "{query}"

Spark UI Data (All Tabs):
--- START TEXT ---
{truncated_text}
--- END TEXT ---

Analyze the above data specifically for {case_name} issues. Provide evidence-based findings.
""")
        ]
        
        try:
            findings, metrics = track_llm_call(agent_name, llm, prompt_messages, f"Detecting {case_name}")
        except Exception as e:
            print(f"Error in {agent_name} LLM call: {e}")
            findings = f"Error analyzing {case_name}: {e}"
            metrics = {
                "total_tokens": 0,
                "total_latency": 0.0,
                "llm_calls": [{
                    "agent": agent_name,
                    "description": f"Detecting {case_name} (ERROR)",
                    "latency": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "error": str(e),
                    "timestamp": time.time()
                }]
            }

        return {"case_findings": {agent_name: findings}, "metrics": metrics}

    return case_agent

# Modified Case Synthesizer with tracking
def case_synthesizer(state: CaseAgentState, llm: BaseChatModel) -> Dict[str, Any]:
    print("--- Running Case Synthesizer ---")
    query = state["user_query"]
    history = state["conversation_history"]
    formatted_history = format_history_for_prompt(history, max_turns=8)
    findings = state["case_findings"]

    if not findings:
        analysis = "I could not detect any specific performance anti-patterns in your Spark application based on the provided data. The application appears to be running normally, or the symptoms of common issues are not present in the current dataset."
        metrics = {"total_tokens": 0, "total_latency": 0.0, "llm_calls": []}
        return {"final_analysis": analysis, "metrics": metrics}

    findings_str = "\n\n".join([f"--- {agent_name.replace('_agent', '')} Detection Results ---:\n{result}"
                                for agent_name, result in findings.items()])

    prompt_messages = [
        SystemMessage(content="""
You are a Spark performance optimization expert specializing in anti-pattern detection and remediation.
Your task is to synthesize case-specific findings into actionable recommendations.

Focus on providing:
1. Clear identification of detected performance anti-patterns
2. Evidence from the Spark UI data supporting each detection
3. Specific configuration changes or code modifications to fix issues
4. Priority ranking of issues based on their performance impact

Be concrete and actionable. If no issues are detected, state that clearly.
"""),
        HumanMessage(content=f"""
Conversation History:
{formatted_history}

Original User Query: "{query}"

Case Detection Results:
{findings_str}

Based on the case analysis results, provide a comprehensive performance assessment with specific recommendations.
""")
    ]
    
    try:
        analysis, metrics = track_llm_call("case_synthesizer", llm, prompt_messages, "Synthesizing case findings")
    except Exception as e:
        print(f"Error in Case Synthesizer LLM call: {e}")
        analysis = f"Error synthesizing the case analysis: {e}"
        metrics = {
            "total_tokens": 0,
            "total_latency": 0.0,
            "llm_calls": [{
                "agent": "case_synthesizer",
                "description": "Synthesizing case findings (ERROR)",
                "latency": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error": str(e),
                "timestamp": time.time()
            }]
        }

    return {"final_analysis": analysis, "metrics": metrics}

# Case Response Formatter (no LLM call, no tracking needed)
def case_response_formatter(state: CaseAgentState) -> Dict[str, Any]:
    print("--- Running Case Response Formatter ---")
    analysis = state.get("final_analysis", "No analysis available.")
    formatted_response = format_response(analysis)
    return {"final_response": formatted_response}

# Enhanced metrics display with better error handling
def display_metrics_summary(metrics: Dict[str, Any]):
    """Display a summary of token usage and latency metrics."""
    print("\n" + "="*50)
    print("📊 PERFORMANCE METRICS SUMMARY")
    print("="*50)
    
    # Safely get metrics with defaults
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
    
    # Show any errors in metrics themselves
    if "error" in metrics:
        print(f"\n⚠️  Session Error: {metrics['error']}")
    
    print("="*50)

# Modified session management with metrics
def run_case_chat_turn(session_id: str, user_query: str, pdf_texts: Optional[Dict[str, str]] = None):
    global session_storage
    
    # Track overall execution time
    overall_start_time = time.time()
    
    if session_id not in session_storage:
        if pdf_texts is None: 
            return "Error: PDF texts must be provided for the first turn of a session."
        print(f"Initializing new case analysis session: {session_id}")
        session_storage[session_id] = { "conversation_history": [], "pdf_texts": pdf_texts }
    elif pdf_texts is not None:
        print(f"Warning: PDF texts provided for existing session {session_id}. Overwriting.")
        session_storage[session_id]["pdf_texts"] = pdf_texts

    current_session = session_storage[session_id]
    current_history = current_session["conversation_history"]
    current_pdfs = current_session["pdf_texts"]

    graph_input: CaseAgentState = {
        "session_id": session_id,
        "user_query": user_query,
        "pdf_texts": current_pdfs,
        "conversation_history": current_history.copy(),
        "cases_to_check": [],
        "case_findings": {},
        "metrics": {"total_tokens": 0, "total_latency": 0.0, "llm_calls": []},
        "final_analysis": None,
        "final_response": None,
    }

    print(f"\n--- Invoking Case Analysis Graph for Session {session_id} ---")
    try:
        final_state = app.invoke(graph_input)
        agent_response = final_state.get("final_response", "Sorry, I encountered an issue processing your request.")
        
        overall_end_time = time.time()
        overall_execution_time = overall_end_time - overall_start_time
        
        print("--- Case Analysis Graph Invocation Complete ---")
        
        # Display metrics
        metrics = final_state.get("metrics", {})
        metrics["overall_execution_time"] = overall_execution_time
        display_metrics_summary(metrics)
        
    except Exception as e:
        overall_end_time = time.time()
        overall_execution_time = overall_end_time - overall_start_time
        
        print(f"--- Case Analysis Graph Invocation ERROR: {e} ---")
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

# Graph Definition
workflow = StateGraph(CaseAgentState)

# Add nodes
workflow.add_node("case_orchestrator", partial(case_orchestrator, llm=llm))

# Create case agents
case_names = [
    'skewed_join', 'huge_collect', 'too_many_partitions', 'cache_no_unpersist',
    'python_udf', 'cartesian_join', 'many_small_files', 'no_compression',
    'multi_cache', 'rdd_conversion', 'autoscaling_backlog', 'broadcast_threshold',
    'memory_spill', 'pandas_udf', 'gc_heavy_rdd'
]

case_agent_names = {}
for case_name in case_names:
    agent_name = f"{case_name}_agent"
    case_agent_names[case_name] = agent_name
    workflow.add_node(agent_name, create_case_agent(case_name, llm))

workflow.add_node("case_synthesizer", partial(case_synthesizer, llm=llm))
workflow.add_node("case_formatter", case_response_formatter)

# Define edges
workflow.set_entry_point("case_orchestrator")

def route_to_case_agents(state: CaseAgentState):
    agent_names_to_call = state.get("cases_to_check", [])
    if not agent_names_to_call:
        print("Routing: Case Orchestrator -> Case Synthesizer (no cases needed)")
        return "case_synthesizer"
    else:
        print(f"Routing: Case Orchestrator -> Case Agents ({agent_names_to_call})")
        return agent_names_to_call

workflow.add_conditional_edges(
    "case_orchestrator",
    route_to_case_agents,
    {
        "case_synthesizer": "case_synthesizer",
        **{agent_name: agent_name for agent_name in case_agent_names.values()}
    }
)

for agent_name in case_agent_names.values():
    workflow.add_edge(agent_name, "case_synthesizer")

workflow.add_edge("case_synthesizer", "case_formatter")
workflow.add_edge("case_formatter", END)

# Compile the graph
print("Compiling Case-Based LangGraph application...")
try:
    app = workflow.compile()
    print("Graph compiled successfully.")
except Exception as e:
    print(f"Error compiling graph: {e}")
    exit()

# Session Management
session_storage = {}

# Main execution
if __name__ == "__main__":
    pdf_folder = "runs/6 - GC Heavy RDD Demo" 
    print(f"Attempting to load PDFs from: {os.path.abspath(pdf_folder)}")

    try:
        actual_pdf_texts = load_pdf_texts_from_folder(pdf_folder)
    except Exception as load_err:
        print(f"FATAL ERROR: Could not load PDF files: {load_err}")
        actual_pdf_texts = {}

    if not actual_pdf_texts:
        print("\nERROR: No PDF data was loaded. Cannot start case analysis session.")
        print("Please ensure the PDF files exist in the specified folder and are readable.")
        print("Expected filenames: environment.pdf, executors.pdf, jobs.pdf, sql.pdf, stages.pdf, storage.pdf")
        exit()
    else:
        print("\nPDF Loading Summary:")
        for key, text in actual_pdf_texts.items():
            print(f"  - Loaded '{key}': {len(text)} characters")
        print("-" * 30)

    my_session_id = str(uuid.uuid4())
    print(f"Initializing new case analysis session: {my_session_id}")
    session_storage[my_session_id] = {
        "conversation_history": [],
        "pdf_texts": actual_pdf_texts
    }

    print("\nCase-based Spark Performance Analysis session started.")
    print("Ask questions about potential performance issues in your Spark application.")
    print("The system will detect specific anti-patterns and provide targeted recommendations.")
    print("Type 'quit' or 'exit' to end the session.")
    print("-" * 30)

    while True:
        try:
            user_query = input("You: ")
        except EOFError:
            print("\nExiting...")
            break

        if user_query.lower().strip() in ["quit", "exit"]:
            print("Assistant: Goodbye!")
            break

        if not user_query.strip():
            continue

        assistant_response = run_case_chat_turn(my_session_id, user_query)
        print(f"Assistant: {assistant_response}")
        print("-" * 10)

    print("\nCase analysis session ended.")