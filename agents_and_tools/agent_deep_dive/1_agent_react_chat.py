import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Define tools
def get_current_time(*args, **kwargs):
    import datetime
    return datetime.datetime.now().strftime("%I:%M %p")

def search_wikipedia(query):
    from wikipedia import summary
    try:
        return summary(query, sentences=2)
    except:
        return "I'm sorry, I couldn't find any information on that topic."

# Define the tools that the agent can use
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Get the current time",
    ),
    
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Search Wikipedia for information",
    ),
]

# Load the prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o")

# Create a structed chat agent with conversation buffer memory
# It stores the conversation history and can be used to generate more context-aware responses
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# Create structured chat agent
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat loop
while True:
    user_input = input("User:")
    if user_input.lower() == "exit":
        break
    
    # Add user message to memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    
    # Invoke the agent
    response = agent_executor.invoke({"input":user_input})
    print("Bot:", response["output"])

    # Add bot message to memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))