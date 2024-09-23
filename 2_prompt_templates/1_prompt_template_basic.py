from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# # P1: Create a ChatPromptTemplate using a template string
# template = 'Tell me a joke about {topic}.'
# prompt_template = ChatPromptTemplate.from_template(template)

# # Print the prompt template
# prompt = prompt_template.invoke({"topic": "cats"})
# print(prompt)


# P2: Prompt with multiple Placeholders
# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:"""

# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "cat"})
# print(prompt)

# P3: Prompt with System and Human Messages (Using tuples)
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}"),
#     ("human", "Tell me {joke_count} jokes."),
# ]

# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "cats", "joke_count": 3})
# print(prompt)

# Extra Information about P3:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    (HumanMessage(content="Tell me 3 jokes.")),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "cats", "joke_count": 3})
print(prompt)

# # This does NOT work:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me {joke_count} jokes."),
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
# print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
# print(prompt)