import os
from typing import Type

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

load_dotenv()

# Pydantic models and tool arguments

class SimpleSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    

class MultipleNumbersArgs(BaseModel):
    x: float = Field(description="The first number to multiply.")
    y: float = Field(description="The second number to multiply.")
    

# Custom tool with only custom input

class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "useful for when you need to answer questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput
    
    def _run(self, query: str) -> str:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key)
        response = client.search(query)
        
        return f"Search results for: {query}\n\n\n{response}\n\n"
    
class MultipleNumbersTool(BaseTool):
    name = "MultipleNumbers"
    description = "Multiplies two numbers."
    args_schema: Type[BaseModel] = MultipleNumbersArgs
    
    def _run(self, x: float, y: float) -> str:
        result = x * y
        return f"The result of multiplying {x} and {y} is {result}."
    

tools = [
    SimpleSearchTool(),
    MultipleNumbersTool(),
]

# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o")

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the ReAct agent using the create_tool_calling_agent function
agent = create_tool_calling_agent(
    llm=llm,  # Language model to use
    tools=tools,  # List of tools available to the agent
    prompt=prompt,  # Prompt template to guide the agent's responses
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Search for Apple Intelligence"})
print("Response for 'Search for LangChain updates':", response['output'])

response = agent_executor.invoke({"input": "Multiply 10 and 20"})
print("Response for 'Multiply 10 and 20':", response['output'])