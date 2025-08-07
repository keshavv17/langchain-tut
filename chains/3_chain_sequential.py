from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")

#define prompt templates
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts"),
    ]
)

#define a prompt template for translation to french
translation_template = ChatPromptTemplate.from_messages([
    ("system","You are a translator and convert the following text to {language}"),
    ("human", "translate the following text to {language}:{text}")
])

# define additional processing steps using runnable lambda 
count_words = RunnableLambda(lambda x: f"word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})

# create combined chain using lcel
chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

# run the chain
result = chain.invoke({"animal": "cat", "fact_count": 2})

print(result)