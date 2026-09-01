from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate   
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

# Components

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Two different prompts to run in parallel
prompt1 = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)
prompt2 = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

# Input 

topic = "Retrival Augmented Generation (RAG)"

chain = RunnableParallel({

    "simple": RunnableLambda(
    lambda x: {"topic": x['simple']}    # Take x as input → return {"topic": x["simple"]}
    ) | prompt1 | model | parser,

    "detailed": RunnableLambda(
    lambda x: {"topic": x['detailed']}  # Take x as input → return {"topic": x["detailed"]}
    ) | prompt2 | model | parser
})

result = chain.invoke({
    "simple" : "Retrival Augmented Generation (RAG)",
    "detailed" :"Agentic AI "
})
print(result['simple'])
print(result['detailed'])
