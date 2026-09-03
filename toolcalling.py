from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_mistralai import ChatMistralAI
from rich import print
from langchain_core.messages import HumanMessage
from langchain.tools import tool

# create a custom tool using the @tool decorator

@tool
def get_text_length(text: str) -> int: # type hinting for the input and output
    """
    Return the length of characters in the given text.
    """
    return len(text)

tools = {
    "get_text_length" :get_text_length
}

llm = ChatMistralAI(model="mistral-small-latest")

# bind the tool to the llm instance

llm_with_tool = llm.bind_tools([get_text_length])

message = []

prompt = input("You: ")
query = HumanMessage(prompt)
message.append(query)

result = llm_with_tool.invoke(message)

message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

    result = llm_with_tool.invoke(message) 
    print(result.content)
    