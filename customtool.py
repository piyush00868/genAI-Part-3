from langchain.tools import tool

@tool # decorator for creating a tool
def get_greeting(name: str) -> str:  # type hinting for the input and output
    """
    Generate a greeting message for the given name."""  # docstring for the tool
    return f"Hello {name}, welcome to AI world"

result = get_greeting.invoke({"name": "Piyush"})  # invoking the tool with input parameters
print(result)

print(get_greeting.name)  # prints the name of the tool
print(get_greeting.description)  # prints the description of the tool
print(get_greeting.args)  # prints the input arguments of the tool