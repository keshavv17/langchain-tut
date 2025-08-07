from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages = [
    SystemMessage("You're an expert in SEO"),
    HumanMessage("Give a short tip on creating engaging content")
]

result = llm.invoke(messages)

print(result.content)