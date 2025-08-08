import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
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
    
    # read the text contents from the file
    loader = TextLoader(file_path)
    documents = loader.load();

    # split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size = 768, chunk_overlap = 50)
    docs = text_splitter.split_documents(documents)

    # create embeddings
    print("\n---- Creating embeddings ----")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )  
    print("\n---- Finished creating embeddings ----")

    # create the vector store and persist it automatically
    print("\n---- Creating vector store ----")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n---- Finished Creating vector store ----")

else:
    print("Vector store already exists. No need to initialize.")