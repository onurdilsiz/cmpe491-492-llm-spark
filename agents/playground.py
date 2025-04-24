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


web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    # Store the agent sessions in a sqlite database
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
    # Adds the current date and time to the instructions
    add_datetime_to_instructions=True,
    # Adds the history of the conversation to the messages
    add_history_to_messages=True,
    # Number of history responses to add to the messages
    num_history_responses=5,
    # Adds markdown formatting to the messages
    markdown=True,
)

code_quality_agent = Agent(
    name="Code Quality Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "If any code is provided, analyze it according to the following instruction, and if not given and not found in the context , please ask for the code.",
        PROMPT
        ],
    storage=SqliteAgentStorage(table_name="code_quality_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

crawler_agent = Agent(
    name="Crawler Agent",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "If any link is provided crawl the link and provide the information according to the following instruction, and if not given and not found in the context , please ask for the link.",
        ],
    tools=[FirecrawlTools(scrape=False, crawl=True)],
    show_tool_calls=True,  
    storage=SqliteAgentStorage(table_name="crawler_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

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
      instructions="Analyze the Spark UI Jobs tab and identify performance issues in job execution.", 
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True, )






sparky_team = Team(
    name="Sparky Team",
    mode="route",
    model=OpenAIChat("gpt-4o"),
    members=[code_quality_agent, jobs_agent],
    show_tool_calls=True,
    markdown=True,
    description="You are a multi-agent team coordinator for analyzing Apache Spark performance using Spark UI components.",
    instructions=[
        "Identify the purpose of the user's question and direct it to the appropriate sparky agent.",
        "If any screenshot or pdf  is provided, direct it to the jobs agent.",
        "If any code is provided, direct it to the code quality agent.",
        "Remember: You are the final gatekeeper before the analysis that is going to provided, wait for the members to respond and provide the final response.",
    ],
    show_members_responses=True,
    debug_mode=True,

)


app = Playground(teams=[sparky_team]).get_app()

@app.get("/")
async def root():
    return {"message": "Welcome to Agno Playground!"}

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)