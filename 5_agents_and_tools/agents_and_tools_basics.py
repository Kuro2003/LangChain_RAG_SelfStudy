import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# List of tools
tools = [
    Tool(
        name="Time",
        func= get_current_time,
        description="Get the current time",
    ),
]

# Pull the prompt template from the hub
# ReAct = reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# Initialize a ChatOpenAI agent
llm = ChatOpenAI(
    model = "gpt-4o",
    temperature=0
)

# Create a React agent using the create_react_agent function
agent = create_react_agent(
    llm =llm,
    tools = tools,
    prompt=prompt,
    stop_sequence=True # Stop model continue when finishing the prompt
)

# Create an AgentExecutor and run the agent
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True, # Show the details output of each tool
)

# Run the agent
response = agent_executor.invoke({"input": "What is the current time?"})

# Print the response
print(response)


