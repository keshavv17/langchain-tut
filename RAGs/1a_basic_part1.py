import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community import document_loaders
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# check if the chrome vector store alraedy exists
if not os.path.exists(persistent_directory):
    print("persistent directory does not exist. Initializing vector store...")

    # ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path"
        )