from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")

template = "write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max."

prompt_template = ChatPromptTemplate.from_template(template);

prompt = prompt_template.invoke({
    "tone":"energetic",
    "company":"samsung",
    "position":"SDE",
    "skill":"AI"
})

result = llm.invoke(prompt)

print(result.content)