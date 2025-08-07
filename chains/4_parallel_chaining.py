from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")

# define prompt template for movie summary
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a movie critic"),
        ("human","provide a brief summary of {movie_name}.")
    ]
)

# define plot analysis
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a movie critic"),
            ("human", "analyse the plot: {plot}. what are its strengths and weaknesses?")
        ]
    )
    return plot_template.format_prompt(plot = plot)

# define character analysis step
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a movie critic"),
            ("human","analyze the characters: {characters}. What are their strengths and weaknesses")
        ]
    )
    return character_template.format_prompt(characters=characters)

# combine analysis in a final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot analysis:\n{plot_analysis}\n\ncharacter_analysis:\n{character_analysis}"

# simplify branches with lcel
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

# create the combined chain using langchain expression language LCEL
chain = (
    summary_template 
    | model
    | StrOutputParser()
    | RunnableParallel(branches = {"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

# run the chain
result = chain.invoke({"movie_name": "Inception"})

print(result)
