"""
load document in chroma vector database (default) using embedding model
"""
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_document_into_database(model_name: str, documents_path: str, reload: bool, storage: str) -> Chroma:
    """load_docuemnt_into_database

    Args:
        model_name (str): name of embedding model
        docuements_path (str): path to the directory containing all documents
        reload (bool, optional): Whether load documents in this run. Defaults to True.
        storage (str): path to the dir containing database
    Returns:
        Chroma: the vector database
    """
    PERSIST_DIRECTORY = storage
    if reload:
        # reload
        documents = load_documents(documents_path)
        # split
        chunks = TEXT_SPLITTER.split_documents(documents)

        # embedding and store
        print(f"\nstore documents in chroma database, located in {PERSIST_DIRECTORY}\n")
        
        # 尝试不同的ollama服务地址
        base_urls = [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://0.0.0.0:11434"
        ]
        
        for base_url in base_urls:
            try:
                embedding = OllamaEmbeddings(model=model_name, base_url=base_url)
                # 测试连接
                embedding.embed_query("test")
                print(f"成功连接到ollama嵌入服务: {base_url}")
                return Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding,
                    persist_directory=PERSIST_DIRECTORY
                )
            except Exception:
                continue
        
        raise Exception("无法连接到ollama嵌入服务，请检查ollama是否正在运行")
    else:
        print(f"\nread documents in chroma database, located in {PERSIST_DIRECTORY}\n")
        # read
        return Chroma(
            embedding_function=OllamaEmbeddings(model=model_name),
            persist_directory=PERSIST_DIRECTORY
        )

def load_documents(documents_path:str) -> list[Document]:
    """
    Load documents under a directory

    Args:
        documents_path (str): _description_

    Returns:
        list[Document]: _description_
    """
    loaders = {
        ".pdf": DirectoryLoader(
            path=documents_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path=documents_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True
        )
    }
    
    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs