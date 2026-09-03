from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

load_dotenv()

search_tool = TavilySearch(
    max_results=5,
)

model = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    '''
You are a helpful assistant that summarize the following news into clear bullet points.
{news}
'''
)

chain = prompt | model | StrOutputParser()

news_result = search_tool.invoke("Latest news about AI")

print("\n===== SOURCES =====\n")

for result in news_result["results"]:
    print("Title:", result["title"])
    print("URL:", result["url"])
    print()


result = chain.invoke({"news": news_result})


print("\n===== SUMMARY =====\n")

print(result)
