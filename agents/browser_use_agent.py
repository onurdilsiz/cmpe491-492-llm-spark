from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI

import asyncio
# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        chrome_instance_path='C:\Program Files\Google\Chrome\Application\chrome.exe',  
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
    )
)

#  "Go to https://pqbmrig4hndx5dwgt43tb3pq7a-dot-europe-west4.dataproc.googleusercontent.com/gateway/default/sparkhistory/history/app-20250219140302-0000/jobs/ "
#         "and find me the longest job amongst all jobs and go to related job page and give me details about it."

prompt = (
    "Access the Spark UI on https://pqbmrig4hndx5dwgt43tb3pq7a-dot-europe-west4.dataproc.googleusercontent.com/gateway/default/sparkhistory/history/app-20250219140302-0000/jobs/ :\n"
    "Navigate to the Spark UI Executors tab. Identify key metrics related to executor usage, such as the number of active executors over time, task completion times, and any periods of executor idleness.\n"
    "Analyze Executor Usage:\n"
    "Examine the executor usage patterns throughout the application lifecycle. Identify periods where executors were underutilized or where there was a backlog in the task queue.\n"
    "Determine Initial Executors:\n"
    "Based on the initial task demand and startup time, assess whether the current initialExecutors setting is sufficient. If the application experiences delays at startup, consider increasing this value to match the initial workload.\n"
    "Evaluate Dynamic Allocation Ratio:\n"
    "Review the task backlog and executor allocation during peak demand. If the backlog frequently exceeds the available executors, consider adjusting the dynamicAllocationRatio to allocate more executors during high-demand periods.\n"
    "Adjust Min and Max Executors:\n"
    "Determine the minimum and maximum number of executors needed to handle the workload efficiently. If the task backlog is consistently high, suggest increasing maxExecutors. Ensure minExecutors is set to maintain a baseline level of performance."
)

prompt2 = (
    "Access the Spark UI on https://pqbmrig4hndx5dwgt43tb3pq7a-dot-europe-west4.dataproc.googleusercontent.com/gateway/default/sparkhistory/history/app-20250219140302-0000/jobs/ :\n"
    "Review Current Configuration:\n"
    "Check the current setting of spark.sql.autoBroadcastJoinThreshold on Environment tab under the spark properties, you should use search or scroll down to find it. If it is set to -1, broadcast joins are disabled. Consider setting it to a default value around 200 MB to enable broadcast joins for smaller datasets.\n"
    "Analyze Data Characteristics:\n"
    "Examine the sizes of the data partitions involved in join operations. Identify any partitions that are slightly larger than the current threshold and could benefit from broadcast joins.\n"
    "Provide Suggestions:\n"
    "If you notice that certain partition sizes are just above the current threshold (e.g., 240 MB), suggest increasing the spark.sql.autoBroadcastJoinThreshold slightly to accommodate these sizes. This can enable broadcast joins for these partitions, potentially improving join performance.\n"
    "Explain Potential Benefits:\n"
    "Highlight the potential performance improvements from enabling broadcast joins for partitions that fit within the adjusted threshold. Explain how this can reduce shuffle operations and improve execution speed.\n"
    "Encourage Testing and Validation:\n"
    "Recommend testing the suggested threshold adjustment by running the Spark job and monitoring performance metrics. Use the Spark UI to verify that broadcast joins are being utilized effectively.\n"
    "Document and Iterate:\n"
    "Advise documenting the changes and their impact on job performance. Encourage iterative adjustments based on observed results and data characteristics."
)

agent = Agent(
    task=( "go to https://pqbmrig4hndx5dwgt43tb3pq7a-dot-europe-west4.dataproc.googleusercontent.com/gateway/default/sparkhistory/history/app-20250219140302-0000/jobs/ click duration twice to sort them decreasingly, save the page as pdf" ),
    llm= ChatOpenAI(model="gpt-4o"),
    browser=browser
)

async def main():
  
    await agent.run()
    input('Press Enter to close the browser...')
    await browser.close()

asyncio.run(main())