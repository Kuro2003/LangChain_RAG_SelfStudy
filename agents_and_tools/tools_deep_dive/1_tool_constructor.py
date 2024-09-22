from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent 
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI

# Functions for the tools
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"


def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b

# Pydantic model for the tool arguments
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="The first string.")
    b: str = Field(description="The second string.")
    

tools = [
    # Use Tool for simpler functions with a single input parameter
    # This is straightforward and easy to use
    Tool(
        name="Greet User",
        func=greet_user,
        description="Greets the user by name.",
    ),
    Tool(
        name="Reverse String",
        func=reverse_string,
        description="Reverses the given string.",
    ),
    
    StructuredTool.from_function(
        name="Concatenate Strings",
        func=concatenate_strings,
        description="Concatenates two strings.",
        args_model=ConcatenateStringsArgs,
    ),
]

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o")

# Pull the prompt from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the ReAct agent using the create_tool_calling_agent function
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse Hello"})
print("Response for 'Reverse Hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world' "})
print("Response for 'Concatenate Strings':", response)