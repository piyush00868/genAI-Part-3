import os
import requests

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print


load_dotenv()


# ==========================================
# 1. WEATHER TOOL
# ==========================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric", 
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return f"Could not get weather information for {city}."

    data = response.json()

    temperature = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]

    return (
        f"Weather in {city}:\n"
        f"Temperature: {temperature}°C\n"
        f"Feels like: {feels_like}°C\n"
        f"Humidity: {humidity}%\n"
        f"Condition: {description}"
    )


# ==========================================
# 2. TAVILY NEWS TOOL
# ==========================================

search_tool = TavilySearch(
    max_results=5,
    topic="news"
)


# ==========================================
# 3. CREATE TOOL LIST
# ==========================================

tools = [
    get_weather,
    search_tool
]


# ==========================================
# 4. CREATE TOOL MAP
# ==========================================

tool_map = {
    get_weather.name: get_weather,
    search_tool.name: search_tool
}

# print("Available tools:")
# print(tool_map.keys())


# ==========================================
# 5. GROQ MODEL
# ==========================================

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

llm_with_tools = model.bind_tools(tools)


# ==========================================
# 6. USER INPUT
# ==========================================

prompt = input("You: ")

messages = [
    HumanMessage(content=prompt)
]


# ==========================================
# AGENT LOOP
# ==========================================

while True:

    response = llm_with_tools.invoke(messages)

    # Add Groq response to conversation
    messages.append(response)

    # --------------------------------------
    # No tool call = final answer
    # --------------------------------------

    if not response.tool_calls:

        print("\n" + "=" * 50)
        print("🤖 AI")
        print("=" * 50)
        print(response.content)

        break

    # --------------------------------------
    # Execute tools
    # --------------------------------------

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # print("\n" + "-" * 50)
        # print(f"🔧 Using tool: {tool_name}")
        # print("-" * 50)

        tool = tool_map.get(tool_name)

        if tool is None:

            tool_result = f"Tool {tool_name} not found."

        else:

            tool_result = tool.invoke(tool_args)

        # ----------------------------------
        # Display clean tool output
        # ----------------------------------

        if tool_name == "get_weather":

            print(tool_result)

        elif tool_name == "tavily_search":

            print("📰 Latest news retrieved successfully.")

        # ----------------------------------
        # Send result back to Groq
        # ----------------------------------

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )