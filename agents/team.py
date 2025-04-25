from pathlib import Path
from pypdf import PdfReader                       # pip install pypdf

def extract_pdf_text(path: str | Path,
                     max_chars: int | None = 12_000) -> str:
    """
    Read a (small) PDF and return its plaintext.
    If `max_chars` is set, the text is truncated to keep the prompt short.
    """
    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated …]"
    return text


from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.firecrawl import FirecrawlTools
from agno.team.team import Team



PROMPT ="""    Analyze the provided Spark code and detect opportunities to improve performance, resource utilization, and maintainability by addressing the following best practices. For each detected case:

    **DataFrame/Dataset over RDD:**

    Identify areas where RDDs are used and provide their location in the code (e.g., line number or code snippet).
    Suggest converting them to DataFrame/Dataset with an equivalent transformation or action.
    Explain the benefits, such as query optimizations, reduced shuffling, and easier integration with structured data formats.
    **coalesce() over repartition():**

    Detect instances where repartition() is used and provide its location in the code (e.g., line number or code snippet).
    Evaluate whether the operation requires a full shuffle or if reducing partitions suffices.
    Suggest replacing repartition() with coalesce() where applicable and provide an equivalent code example.
    Highlight the benefits of switching to coalesce(), such as reduced shuffling, improved resource usage, and faster job runtime.
    **mapPartitions() over map():**

    Identify where map() transformations are used and provide their location in the code (e.g., line number or code snippet).
    Evaluate whether the transformation can be performed at the partition level (e.g., batching or I/O-heavy operations).
    Suggest replacing map() with mapPartitions() where applicable and provide an equivalent code example.
    Highlight the benefits of switching to mapPartitions(), such as reduced function call overhead, optimized I/O, and improved performance for partition-level operations.
    **Serialized Data Formats:**

    Check for input/output operations using non-optimized data formats (e.g., CSV, JSON) and provide their location in the code (e.g., line number or code snippet).
    Suggest switching to optimized formats like Parquet, ORC, or Avro, and provide an equivalent code example.
    Explain the benefits of using serialized formats, such as faster reads/writes, compression, and query optimization through predicate pushdown.
    **Avoiding UDFs:**

    Identify User-Defined Functions (UDFs) in the code and provide their location (e.g., line number or code snippet).
    Evaluate whether the UDF can be replaced with a Spark SQL function or native DataFrame/Dataset operation.
    Suggest the alternative implementation with a code example.
    Explain the benefits of avoiding UDFs, such as enabling Catalyst optimizations, improving performance, and reducing serialization overhead.

    **Input Code:**
    ```python
    {code}
    ```


    Output Requirements: The response must strictly follow JSON formatting without any additional text, annotations, or explanations. The JSON object must include the following fields:

    "detected0": A boolean indicating whether RDD usage was detected in the provided code.

    "occurrences0": The number of times RDDs are used in the code.

    "response0": An array of objects, where each object provides:

    - "operation": The specific RDD operation with its location in the code (e.g., line number or code snippet).
    - "improvementExplanation": A detailed explanation of why and how the RDD can be replaced with DataFrame/Dataset.
    - "dataframeEquivalent": A code snippet showing the equivalent DataFrame/Dataset transformation.
    -"benefits": A summary of the benefits of switching to DataFrame/Dataset.
    - "detected1": A boolean indicating whether repartition() usage was detected in the provided code.

    "occurrences1": The number of times repartition() is used in the code.

    "response1": An array of objects, where each object provides:

    - "operation": The specific repartition() operation with its location in the code.
    - "improvementExplanation": A detailed explanation of why and how repartition() can be replaced with coalesce().
    - "coalesceEquivalent": A code snippet showing how to replace the repartition() operation with coalesce().
    - "benefits": A summary of the benefits of switching to coalesce().

    "detected2": A boolean indicating whether map() usage was detected in the provided code.

    "occurrences2": The number of times map() is used in the code.

    "response2": An array of objects, where each object provides:

    - "operation": The specific map() operation with its location in the code.
    - "improvementExplanation": A detailed explanation of why and how map() can be replaced with mapPartitions().
    - "mapPartitionsEquivalent": A code snippet showing how to replace the map() operation with mapPartitions().
    - "benefits": A summary of the benefits of switching to mapPartitions().
    "detected3": A boolean indicating whether non-optimized data formats were detected in the provided code.

    "occurrences3": The number of times non-optimized data formats are used in the code.

    "response3": An array of objects, where each object provides:

    - "operation": The specific operation using non-optimized data formats with its location in the code.
    - "improvementExplanation": A detailed explanation of why and how the format can be replaced with optimized serialized formats.
    - "optimizedEquivalent": A code snippet showing how to replace the format with Parquet, ORC, or Avro.
    - "benefits": A summary of the benefits of switching to optimized formats.

    "detected4": A boolean indicating whether UDF usage was detected in the provided code.

    "occurrences4": The number of times UDFs are used in the code.

    "response4": An array of objects, where each object provides:

    - "operation": The specific UDF operation with its location in the code.
    - "improvementExplanation": A detailed explanation of why and how the UDF can be replaced with Spark SQL functions or native DataFrame/Dataset operations.
    - "alternativeEquivalent": A code snippet showing the alternative implementation.
    - "benefits": A summary of the benefits of avoiding UDFs.





"benefits": A summary of the benefits of replacing UDFs, such as enabling Catalyst optimizations, improving performance, and reducing serialization overhead.
"""
agent_storage: str = "tmp/agents.db"


agent_prompts = {
    "jobs": """
You are the Jobs Analyzer Agent specialized in reviewing Apache Spark Jobs tab information. Your task is to extract key metrics from the Jobs tab PDF and identify performance bottlenecks.

Input:
- PDF content from the Jobs tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - Total number of jobs
   - Job completion status (successful vs. failed jobs)
   - Job durations (min, max, average)
   - Jobs with exceptionally long duration
   - Jobs with multiple retry attempts
   - Distribution of stages per job
   - Timeline patterns (e.g., sequential vs. parallel job execution)
   - Any errors or warnings reported for jobs

2. Analyze for these common issues:
   - Jobs with unusually long duration compared to others
   - Failed jobs and their error patterns
   - Jobs with a disproportionately high number of stages
   - Sequential job execution that could be parallelized
   - Jobs that were resubmitted multiple times
   - Stages within jobs that account for the majority of execution time

3. For each identified issue, provide:
   - The specific job ID and description
   - Relevant metrics showing the problem
   - Potential root causes
   - Specific recommendations to address the issue

Output your analysis as Markdown with the following sections:
- Summary Statistics: Key metrics about all jobs
- Critical Issues: High-priority bottlenecks demanding immediate attention
- Performance Observations: Notable patterns that might impact performance
- Recommendations: Specific, actionable suggestions for improvement
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Only make claims that are directly supported by the data in the PDF. Avoid speculation beyond what the data shows.

PDF Content:
{pdf_content}
""",
    
    "stages": """
You are the Stages Analyzer Agent specialized in reviewing Apache Spark Stages tab information. Your task is to extract key metrics from the Stages tab PDF and identify performance bottlenecks at the stage level.

Input:
- PDF content from the Stages tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - Total number of stages
   - Stage completion status (successful vs. failed stages)
   - Stage durations (min, median, max)
   - Number of tasks per stage
   - Task metrics including:
     - Task duration (min, median, max)
     - Shuffle read/write sizes
     - Executor run time
     - JVM GC time
     - Serialization/deserialization time
     - Memory spill (disk/memory)
   - Input/output data sizes

2. Calculate and analyze for data skew indicators:
   - Task duration variance within stages
   - Shuffle read/write size variance across tasks
   - Peak-to-median ratio for task durations

3. Identify these common stage-level issues:
   - Stages with significant data skew (max task duration > 2x median)
   - Stages with excessive shuffle operations
   - Stages with high GC overhead (GC time > 20% of execution time)
   - Stages with large data spills to disk
   - Stages with serialization bottlenecks
   - Stages with too few or too many tasks
   - Sequential stages that could be parallelized

4. For each identified issue, provide:
   - The specific stage ID and description
   - Relevant metrics showing the problem
   - Potential root causes
   - Specific recommendations to address the issue

Output your analysis as Markdown with the following sections:
- Summary Statistics: Key metrics about all stages
- Critical Issues: High-priority bottlenecks demanding immediate attention
- Data Skew Analysis: Evidence of unbalanced data distribution
- Shuffle Operation Analysis: Assessment of data movement operations
- Recommendations: Specific, actionable suggestions for improvement
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Only make claims that are directly supported by the data in the PDF. Focus particularly on quantifying data skew since this is a common Spark performance killer.

PDF Content:
{pdf_content}
""",
    
    "storage": """
You are the Storage Analyzer Agent specialized in reviewing Apache Spark Storage tab information. Your task is to extract key metrics from the Storage tab PDF and identify storage-related performance issues.

Input:
- PDF content from the Storage tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - RDDs/DataFrames being cached
   - Storage level for each cached item (MEMORY_ONLY, MEMORY_AND_DISK, etc.)
   - Memory usage per RDD/DataFrame
   - Disk usage per RDD/DataFrame (if applicable)
   - Partition counts for cached data
   - Cache hit/miss ratios (if available)
   - Fraction of cached data (vs. total size)
   - Replication factors

2. Analyze for these common storage-related issues:
   - Inefficient storage levels for workload patterns
   - Excessive caching that could lead to memory pressure
   - Under-utilization of cache (low cache hit ratio)
   - Uneven partition sizes in cached data
   - Large objects being spilled to disk
   - Objects that are cached but never reused
   - Missing cache opportunities for frequently reused data

3. For each identified issue, provide:
   - The specific RDD/DataFrame ID and description
   - Storage level and size metrics
   - Why this represents a potential problem
   - Specific recommendations to address the issue

Output your analysis as Markdown with the following sections:
- Summary Statistics: Overview of cache usage
- Memory Usage Analysis: Breakdown of memory utilization
- Storage Efficiency Issues: Identified problems with caching strategy
- Partition Distribution: Analysis of partition sizes and counts
- Recommendations: Specific, actionable suggestions for improvement
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Focus on whether the caching strategy is appropriate for the workload and identify opportunities to optimize memory usage while maintaining performance.

PDF Content:
{pdf_content}
""",
    
    "environment": """
You are the Environment Analyzer Agent specialized in reviewing Apache Spark Environment tab information. Your task is to extract configuration settings from the Environment tab PDF and identify potentially suboptimal configurations.

Input:
- PDF content from the Environment tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - Spark version
   - Java/Scala version
   - System specifications (cores, memory)
   - All Spark configuration parameters, with special focus on:
     - spark.executor.memory
     - spark.executor.cores
     - spark.executor.instances
     - spark.driver.memory
     - spark.driver.cores
     - spark.default.parallelism
     - spark.sql.shuffle.partitions
     - spark.memory.fraction
     - spark.memory.storageFraction
     - spark.locality.wait
     - spark.speculation
     - spark.serializer
     - spark.io.compression.codec
     - spark.rdd.compress
     - spark.shuffle.compress
     - spark.sql.adaptive.enabled
     - spark.dynamicAllocation settings

2. Analyze for these common configuration issues:
   - Suboptimal memory allocation (too high/low)
   - Inefficient parallelism settings
   - Memory fraction imbalances
   - Missing performance-critical configurations
   - Inefficient serialization or compression settings
   - Disabled features that could improve performance
   - Configurations that don't align with cluster resources
   - Outdated Spark version with known performance issues

3. For each identified issue, provide:
   - The specific configuration parameter
   - Current setting and recommended setting
   - Rationale for the recommendation
   - Expected performance impact of the change

Output your analysis as Markdown with the following sections:
- Environment Summary: Key version and system information
- Critical Configuration Issues: High-priority settings that need adjustment
- Memory Configuration Analysis: Assessment of memory-related settings
- CPU/Parallelism Analysis: Assessment of CPU utilization and parallelism settings
- I/O and Serialization Analysis: Assessment of data processing efficiency settings
- Recommended Configuration Changes: Complete list of suggested modifications
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Provide concrete configuration recommendations with numerical values where appropriate, explaining the reasoning behind each suggestion.

PDF Content:
{pdf_content}
""",
    
    "executors": """
You are the Executors Analyzer Agent specialized in reviewing Apache Spark Executors tab information. Your task is to extract key metrics from the Executors tab PDF and identify executor-related performance issues.

Input:
- PDF content from the Executors tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - Total number of executors
   - Executor host distribution
   - Per-executor metrics including:
     - Active tasks
     - Failed tasks
     - Complete tasks
     - Total cores
     - Memory usage (peak and current)
     - On-heap/off-heap memory usage
     - Disk usage
     - Duration
     - GC time
     - Input/output
     - Shuffle read/write
   - Driver metrics (treated as a special executor)

2. Analyze for these common executor-related issues:
   - Uneven task distribution across executors
   - Executors with excessive GC time (GC time > 15% of executor uptime)
   - Memory pressure indicators (high memory usage, frequent GC)
   - Executors with unusually high failure rates
   - Resource utilization imbalances
   - Executors with significantly higher/lower throughput
   - Network I/O bottlenecks
   - Disk I/O bottlenecks
   - Driver as a bottleneck (overloaded driver)

3. For each identified issue, provide:
   - The specific executor ID or host
   - Relevant metrics showing the problem
   - Comparisons with other executors to highlight imbalances
   - Specific recommendations to address the issue

Output your analysis as Markdown with the following sections:
- Summary Statistics: Overview of executor fleet
- Resource Utilization Analysis: CPU, memory, and disk usage patterns
- GC Analysis: Garbage collection patterns and issues
- Task Distribution Analysis: Assessment of work distribution
- Critical Executor Issues: Specific executors with problems
- Driver Analysis: Assessment of driver performance
- Recommendations: Specific, actionable suggestions for improvement
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Pay particular attention to resource utilization patterns and identify opportunities for better work distribution or resource allocation.

PDF Content:
{pdf_content}
""",
    
    "sql": """
You are the SQL Analyzer Agent specialized in reviewing Apache Spark SQL/DataFrame tab information. Your task is to extract key metrics from the SQL tab PDF and identify query performance issues.

Input:
- PDF content from the SQL/DataFrame tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - List of executed SQL queries/DataFrame operations
   - Query execution times
   - Query plans showing:
     - Scan operations (table scans, file formats)
     - Join types and conditions
     - Filter operations
     - Aggregation operations
     - Sort operations
     - Exchange (shuffle) operations
     - Project operations
   - Statistics on rows processed at each stage
   - Predicate pushdown information
   - Partition pruning information

2. Analyze for these common SQL performance issues:
   - Cartesian products or expensive joins
   - Missing filter pushdown opportunities
   - Missing partition pruning opportunities
   - Excessive shuffling in query plans
   - Poorly optimized aggregations
   - Inefficient scan patterns
   - Suboptimal join ordering
   - Broadcast join opportunities being missed
   - Sort operations on large datasets
   - Skew in join or aggregation keys
   - Window functions with large partitions
   - Complex/inefficient expressions

3. For each identified issue, provide:
   - The specific query or operation ID
   - The problematic portion of the query plan
   - Why this represents a performance issue
   - SQL or API-level recommendations to improve the query

Output your analysis as Markdown with the following sections:
- Query Overview: Summary of analyzed queries
- Critical Query Performance Issues: High-priority SQL bottlenecks
- Join Operation Analysis: Assessment of join efficiency
- Data Reading Patterns: Analysis of scan operations
- Aggregation & Grouping Analysis: Assessment of data aggregation patterns
- Shuffle Analysis: Impact of data exchange operations
- Query Optimization Recommendations: Specific suggestions for each problematic query
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Where possible, provide specific SQL rewrites or DataFrame API alternatives that would improve performance.

PDF Content:
{pdf_content}
""",
    
    "synthesizer": """
You are the Synthesizer Agent responsible for integrating the findings from all specialist analyzer agents and producing a comprehensive Spark performance analysis report.

Input:
- Analysis reports from all specialist agents (Jobs, Stages, Storage, Environment, Executors, SQL)

Instructions:
1. Read and assimilate the analyses from all specialist agents.

2. Identify cross-cutting performance issues that span multiple areas, such as:
   - Correlation between configuration settings and observed bottlenecks
   - Connection between SQL query patterns and stage/task performance
   - Relationship between storage decisions and executor memory pressure
   - Impact of join strategies on shuffle operations and network I/O
   - Connections between GC pressure and memory configuration
   - How data skew in stages relates to join/groupBy operations in SQL

3. Prioritize identified issues based on:
   - Severity (impact on overall performance)
   - Frequency of occurrence
   - Difficulty of resolution
   - Potential performance improvement

4. Formulate a holistic set of recommendations addressing:
   - Quick wins (simple configuration changes)
   - Medium-effort improvements (query/job restructuring)
   - Strategic changes (data layout, application architecture)

5. Create a comprehensive executive summary that highlights:
   - The most critical bottlenecks
   - Root causes spanning multiple areas
   - Expected performance improvements from recommendations

Output your analysis as a well-structured Markdown report with the following sections:
- Executive Summary: Brief overview of key findings
- Critical Performance Bottlenecks: Prioritized list of the most significant issues
- Cross-Cutting Analysis: Performance issues that span multiple areas
- Root Cause Analysis: Deeper investigation into underlying causes
- Optimization Recommendations:
  - Configuration Changes
  - Application Code Improvements
  - Data Management Strategies
  - Resource Allocation Adjustments
- Implementation Plan: Suggested order for implementing changes
- Expected Outcomes: Projected performance improvements
- Further Investigation: Areas where more data would be beneficial

Ensure recommendations are specific, actionable, and directly tied to observed issues in the data. Prioritize recommendations based on expected impact and implementation difficulty.

Jobs Analysis:
{jobs_analysis}

Stages Analysis:
{stages_analysis}

Storage Analysis:
{storage_analysis}

Environment Analysis:
{environment_analysis}

Executors Analysis:
{executors_analysis}

SQL Analysis:
{sql_analysis}
"""
}




code_quality_agent = Agent(
    name="Code Quality Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "If any code is provided, analyze it according to the following instruction, and if not given and not found in the context , please ask for the code.",
        PROMPT],
    storage=SqliteAgentStorage(table_name="code_quality_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

jobs_agent = Agent( 
    name="Jobs Agent", 
    role="You analyze the Spark UI Jobs tab and identify performance issues in job execution.",
    model=OpenAIChat(id="gpt-4o"), 
    instructions=[agent_prompts["jobs"]], 
    storage=SqliteAgentStorage(table_name="jobs_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True, )

stages_agent = Agent(
    name="Stages Agent",
    role="You analyze the Spark UI *Stages* tab and detect stage-level bottlenecks.",
      model=OpenAIChat(id="gpt-4o"), 
    instructions=[agent_prompts["stages"]],
    storage=SqliteAgentStorage(
        table_name="stages_agent", db_file=agent_storage
    ),
      add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)


sql_agent = Agent( 
    name="SQL Agent", 
    role="You analyze the Spark UI SQL tab and identify performance issues in SQL execution.",
      model=OpenAIChat(id="gpt-4o"), 
      instructions=[agent_prompts["sql"]],
    storage=SqliteAgentStorage(table_name="sql_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True, )

storage_agent = Agent(
    name="Storage Agent",
    role="You analyze the Spark UI *Storage* tab and find caching / memory issues.",
      model=OpenAIChat(id="gpt-4o"), 
    instructions=[agent_prompts["storage"]],
    storage=SqliteAgentStorage(
        table_name="storage_agent", db_file=agent_storage
    ),
   add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

environment_agent = Agent(
    name="Environment Agent",
    role="You inspect the Spark UI *Environment* tab and flag sub-optimal configs.",
      model=OpenAIChat(id="gpt-4o"), 
    instructions=[agent_prompts["environment"]],
    storage=SqliteAgentStorage(
        table_name="environment_agent", db_file=agent_storage
    ),
  add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)


executors_agent = Agent(
    name="Executors Agent",
    role="You analyze the Spark UI *Executors* tab and surface resource-usage issues.",
      model=OpenAIChat(id="gpt-4o"), 
    instructions=[agent_prompts["executors"]],
    storage=SqliteAgentStorage(
        table_name="executors_agent", db_file=agent_storage
    ),
   add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)
sparky_team = Team(
    name="Sparky Team",
    mode="route",
    model=OpenAIChat("gpt-4o"),
    members=[    code_quality_agent,
    jobs_agent,
    stages_agent,
    storage_agent,
    environment_agent,
    executors_agent,
    sql_agent,],
    show_tool_calls=True,
    markdown=True,
    enable_team_history=True,

      description=(
        "Multi-agent team that analyses Apache Spark performance problems from "
        "Spark-UI PDFs/screenshots and application source code."
    ),
    instructions=[
        # ---------- Core routing rules ----------
        "1. Detect the *type* of user input and route to the best specialist:",
        "   • If a Spark-UI PDF / screenshot is attached:",
        "        ▸ If filename contains 'jobs'      → jobs-agent",
        "        ▸ 'stages'                         → stages-agent",
        "        ▸ 'storage'                        → storage-agent",
        "        ▸ 'environment'                    → environment-agent",
        "        ▸ 'executors'                      → executors-agent",
        "        ▸ 'sql' or 'dataframe'             → sql-agent",
        "        ▸ Otherwise ask the user which tab the file represents.",
        "   • If the message contains code          → code-quality-agent",
        "",
        # ---------- Output handling ----------
        "2. Await the specialist’s reply, then as team-leader compile a short, "
        "   clear summary or pass the specialist’s markdown directly to the user.",
        "",
        # ---------- Fall-backs ----------
        "3. If the input doesn’t fit any category, ask the user to clarify "
        "   (e.g., specify Spark-UI tab or attach code).",
    ],
    show_members_responses=True,
    debug_mode=True,

)


def guess_tab_from_name(pdf_path: Path) -> str:
    name = pdf_path.stem.lower()
    for key in ("jobs", "stages", "storage", "environment",
                "executors", "sql", "dataframe"):
        if key in name:
            return key
    return "unknown"


# ---------- 3. one-shot analysis routine ------------------------------
def analyze_pdf_with_team(pdf_file: str | Path,
                          stream: bool = True) -> None:
    pdf_path  = Path(pdf_file)
    tab_name  = guess_tab_from_name(pdf_path)   # -> “jobs” etc.
    pdf_text  = extract_pdf_text(pdf_path)

    # Compose a user-style message that the router can parse
    user_msg = (
        f"This is the Spark-UI *{tab_name}* tab exported as PDF. "
        f"Please analyse it and report the main performance issues.\n\n"
        f"{pdf_text}"
    )
    sparky_team.print_response(user_msg, stream=stream)




if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="One-shot Spark-UI PDF analysis")
    ap.add_argument("pdf", help="Path to a Spark-UI tab PDF (jobs.pdf, stages.pdf …)")
    ap.add_argument("--no-stream", action="store_true",
                    help="Return full string instead of streaming")
    args = ap.parse_args()

    analyze_pdf_with_team(args.pdf, stream=not args.no_stream)