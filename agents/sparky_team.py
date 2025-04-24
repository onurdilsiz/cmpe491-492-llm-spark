from agno.agent import Agent

from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.team.team import Team
from agno.media import File


agent_storage: str = "tmp/agents.db"

code_quality_agent = Agent(
    name="Code Quality Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "If any code is provided, analyze it according to the following instruction, and if not given and not found in the context , please ask for the code.",
        "Analyze the code for potential bugs, performance issues, and best practices. Provide suggestions for improvement.",
        "Provide a summary of the analysis.",],
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
        "If any screenshot is provided, direct it to the jobs agent.",
        "If any code is provided, direct it to the code quality agent.",
        "Remember: You are the final gatekeeper before the analysis that is going to provided, wait for the members to respond and provide the final response.",
    ],
    show_members_responses=True,
    debug_mode=True,

)




if __name__ == "__main__":
    # Ask "How are you?" in all supported languages
    sparky_team.print_response("""import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

states = {"NY":"New York", "CA":"California", "FL":"Florida"}
broadcastStates = spark.sparkContext.broadcast(states)

data = [("James","Smith","USA","CA"),
    ("Michael","Rose","USA","NY"),
    ("Robert","Williams","USA","CA"),
    ("Maria","Jones","USA","FL")
  ]

rdd = spark.sparkContext.parallelize(data)

def state_convert(code):
    return broadcastStates.value[code]

result = rdd.map(lambda x: (x[0],x[1],x[2],state_convert(x[3]))).collect()
print(result)
""", stream=True)  
