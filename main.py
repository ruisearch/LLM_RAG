import argparse
import sys
from langchain_ollama import ChatOllama
from models import check_if_model_is_available
from document_loader import load_document_into_database
from llm import getChatChain


def main(llm_model_name:str, embedding_model_name:str, document_path:str, reload_str:str) ->None:
    """
    1. prepare llm and embedding model;
    2. store document in vector database;
    3. prepare chat
    4. chat
    Args:
        llm_model_name (str): name of llm
        embedding_model_name (str): name of embedding model
        document_path (str): path to the directory containing documents
    """
    try:
        check_if_model_is_available(llm_model_name)
        check_if_model_is_available(embedding_model_name)
    except Exception as e:
        print(e)
        sys.exit()

    if reload_str.lower() == "false":
        reload = False
    else:
        reload = True
    # store document
    try:
        # db = load_document_into_database(model_name=embedding_model_name, documents_path=document_path)
        db = load_document_into_database(model_name=embedding_model_name, documents_path=document_path, reload=reload)
    except Exception as e:
        print(e)
        sys.exit()

    # chat
    llm = ChatOllama(model=llm_model_name)
    chat = getChatChain(llm,db)

    # session
    while True:
        try:
            user_input = input("\n\nPlease enter your question (or type 'exit' to end): ").strip()
            if user_input.lower() == "exit":
                break
            else:
                chat(user_input)
        except KeyboardInterrupt:
            break

def parse_parameters() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM and RAG with Ollama")
    parser.add_argument(
        "-m",
        "--model",
        default="llama3:8b",
        help="The name of the LLM. Defaults to llama3:8b"
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default="nomic-embed-text",
        help="The name of embedding model. Defaults to nomic-embed-text"
    )
    parser.add_argument(
        "-p",
        "--path",
        default="Research",
        help="The path to the directory containing documents to load. Defaults to ./Research/"
    )
    parser.add_argument(
        "-r",
        "--reload",
        default="True",
        help="Whether reload the database. please type True or False. Defaults to True",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_parameters()
    main(args.model, args.embedding_model, args.path, args.reload)