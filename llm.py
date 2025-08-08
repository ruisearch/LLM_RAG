"""
construct a chat session
"""
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import get_buffer_string
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


memory = ConversationBufferMemory(input_key="question", output_key="answer", return_messages=True)
STANDALONE = ChatPromptTemplate.from_messages([
    ("system", """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question."""),
    ("human", """Chat History:
{chat_history}

Follow Up Input:{question}
Standalone question:
"""
)
])

DOCUMENT_TO_STR = PromptTemplate.from_template(template=
"""
Source Document: {source}, Page {page}:\n{page_content}
""")

system_prompt = """You're a helpful research assistant, who answers questions based on provided research documents in a clear way and easy-to-understand way.
If there are no research documents, or the research documents are irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources
IMPORTANT: Provide your answer directly without showing your thinking process or reasoning steps. Give a clear, concise response based on the research provided.
"""
FINAL_QUESTION = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human","""## Research:
{context}

## Question:
{question}""")
])

def getChatChain(llm: ChatOllama, db: Chroma):
    """
    generate the chat session (chat chain):
    1. standalone question
    2. retrive documents
    3. final input
    4. answer
    5. chat

    Args:
        llm (ChatOllama): the llm model
        db (Chroma): the vector db
    """
    # 1. standalone question
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history")
    )
    standalone = {
        "standalone_question":{
            "question":lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"])
        }
        | STANDALONE
        | llm
        | (lambda x:x.content if hasattr(x, "content") else x)
    }

    # 2. retrieve documents
    retriever = db.as_retriever(kwargs={"k":20})
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"]
    }

    # 3. final input
    final_inputs = {
        "context": lambda x:_combine_documents(x["docs"]),
        "question": itemgetter("question")
    }
    
    # 4. answer
    answer = {
        "answer": final_inputs
        | FINAL_QUESTION
        | llm.with_config(callbacks=[StreamingStdOutCallbackHandler])
        | (lambda x:x.content if hasattr(x,"content") else x),
        "docs": itemgetter("docs")
    }

    final_chain = loaded_memory | standalone | retrieved_documents | answer
    # 5. chat
    def chat(question:str):
        if question.lower() == "/clear":
            clear_conversation_history()
        else:
            inputs = {"question":question}
            # invoke
            result = final_chain.invoke(inputs)
            # store memory
            memory.save_context(inputs, {"answer": result["answer"].content if hasattr(result["answer"], "content") else result["answer"]})

    return chat

def _combine_documents(docs: list, String_format=DOCUMENT_TO_STR, \
    separator="\n\n")->str:
    """combine a list of documents to a string

    Args:
        docs (list): list of documents

    Returns:
        str: string contains all documents
    """
    doc_strings = [format_document(doc, String_format) for doc in docs]
    return separator.join(doc_strings)

def clear_conversation_history():
    global memory
    memory.clear()
    print("Conversation history cleared.")