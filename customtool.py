from langchain.tools import tool

@tool # decorator for creating a tool
def get_greeting(name: str) -> str:  # type hinting for the input and output
    """
    Generate a greeting message for the given name."""  # docstring for the tool
    return f"Hello {name}, welcome to AI world"

result = get_greeting.invoke({"name": "John"})  # invoking the tool with input parameters
print(result)
