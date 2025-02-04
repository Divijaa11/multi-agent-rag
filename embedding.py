import os
import shutil
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH, DATA_PATH, MODEL_NAME

def load_documents(data_path):
    documents = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
            docs = loader.load()
        elif file_extension in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        else:
            print(f"Unsupported file format: {filename}")
            continue

        for doc in docs:
            doc.metadata["source"] = filename
        documents.extend(docs)

    print(f"Loaded {len(documents)} document(s).")
    return documents

def create_embeddings():
    documents = load_documents(DATA_PATH)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=500,length_function=len, add_start_index=True)
    chunks=text_splitter.split_documents(documents)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME
    )
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print("Embeddings created and stored in ChromaDB.")

if __name__ == "__main__":
    create_embeddings()