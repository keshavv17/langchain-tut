from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv() 

# setup firebase firestore
PROJECT_ID = "langchain-tut-40255"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history" 

#initialize firestore client
print("Initializing firestore client...")
client = firestore.Client(project = PROJECT_ID)

# initialize firestore chat msg history
print("Initializing firestore chat msg history...")
chat_history = FirestoreChatMessageHistory(
    session_id = SESSION_ID,
    collection = COLLECTION_NAME,
    client = client,
)

print("chat history initialized")
print("current chat history: ", chat_history.messages)

# initialize chat msg
model = ChatGoogleGenerativeAI(model = "gemini-pro")

print("start chatting with ai, enter quit to exit")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
