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
agent = Agent(
        task="Go to https://pqbmrig4hndx5dwgt43tb3pq7a-dot-europe-west4.dataproc.googleusercontent.com/gateway/default/sparkhistory/history/app-20250219140302-0000/jobs/ "
        "and find me the longest job amongst all job and go to related job page and give me details about it.",
        llm=ChatOpenAI(model="gpt-4o"),
        browser=browser
    )

async def main():
  
    await agent.run()

    input('Press Enter to close the browser...')
    await browser.close()

asyncio.run(main())