from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = [] # an empty list to store the messages

# setting an initial system message (optional)
system_msg = SystemMessage(content = "youre a useful ai assistant")
chat_history.append(system_msg) # adding system msg to chat history

# chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content = query)) # add user msg

    # get ai response using history
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content = response))

    print(f"AI: {response}")

print("---message history---")
print(chat_history)
