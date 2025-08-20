import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_kb")

def load_documents():
    docs = []
    # Load .md/.txt quickly
    loader = DirectoryLoader("kb", glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    docs += loader.load()
    loader = DirectoryLoader("kb", glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    docs += loader.load()
    # Load PDFs
    pdf_loader = DirectoryLoader("kb", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    docs += pdf_loader.load()
    return docs

def main():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    vs.persist()
    print(f"Indexed {len(chunks)} chunks into {PERSIST_DIR}")

if __name__ == "__main__":
    main()