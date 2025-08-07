from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  # extracts the content property 
from langchain_google_genai import ChatGoogleGenerativeAI

# load environment variables from .env file
load_dotenv()

# create model
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# define prompt templates(no need of separate runnable chains like we did before two times invoke)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts"),
    ]
)

# create the combined chain using langchain expression language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# run the chain
result = chain.invoke({"animal":"cat", "fact_count": 2})

#output
print(result)