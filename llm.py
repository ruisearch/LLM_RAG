"""
construct a chat session
"""
from operator import itemgetter
import time
import os
import psutil
import re
# from memory_profiler import profile
# from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.messages import get_buffer_string
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# memory = ConversationBufferMemory(input_key="question", output_key="answer", return_messages=True)
# standalone_templete = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

# Chat History:
# {chat_history}

# Follow Up Input: {question}
# Standalone question:"""

# STANDALONE = ChatPromptTemplate.from_messages([
#     ("system", """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question."""),
#     ("human", """Chat History:
# {chat_history}

# Follow Up Input:{question}
# Standalone question:
# """
# )
# ])
# STANDALONE = PromptTemplate.from_template(standalone_templete)

DOCUMENT_TO_STR = PromptTemplate.from_template(
    template= "Source Document: {source}, Page {page}:\n{page_content}"
)

# system_prompt without the priority statememt (combined with method1 in _combine_documents)
system_prompt = """You're a helpful research assistant, who answers questions based on provided research documents in a clear way and easy-to-understand way.
If there are no research documents, or the research documents are irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the brief answer.
If you're unable to answer the question, do not list sources.
"""

# system prompt with the priority statement (combined with method2 in _combine_documents)
# exeriment show it is useless
# system_prompt = """You're a helpful research assistant, who answers questions based on provided research documents in a clear way and easy-to-understand way.

# IMPORTANT: The research documents are provided with priority labels indicating their relevance to the question:
# - [Highest Priority (Most Relevant)]: These documents are most likely to contain the answer
# - [High Priority (Highly Relevant)]: These documents are very relevant to the question
# - [Medium Priority (Moderately Relevant)]: These documents may contain useful information
# - [Low Priority (Less Relevant, Rank X)]: These documents are less likely to be relevant

# When answering:
# 1. PRIORITIZE information from higher priority documents
# 2. Start with information from "Highest Priority" and "High Priority" documents
# 3. Use lower priority documents only to supplement or verify information from higher priority sources
# 4. If higher priority documents provide a complete answer, you may not need to reference lower priority ones

# If there are no research documents, or the research documents are irrelevant to answering the question, simply reply that you can't answer.
# Please reply with just the brief answer.
# If you're unable to answer the question, do not list sources.
# # """

FINAL_QUESTION = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human","""## Research:
{context}

## Question:
{question}""")
])

# # memory instance
# _memory_instance = None

# def _get_memory():
#     """
#     get the memory instance. Create it if not exists
#     """
#     global _memory_instance
#     if _memory_instance is None:
#         _memory_instance = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
#     return _memory_instance

def getChatChain(llm, db: Chroma, debug:bool, local_model: bool):
    """
    generate the chat session (chat chain):
    1. standalone question
    2. retrive documents
    3. final input
    4. answer
    5. chat

    Args:
        llm : the llm model
        db (Chroma): the vector db
        debug (bool): whether print debug information(time cost)
        local_model (bool): Whether the local model
    """
    # do not generate the standalone question. Just use the user input
    # # 1. standalone question
    # memory = _get_memory()

    # loaded_memory = RunnablePassthrough.assign(
    #     chat_history=RunnableLambda(memory.load_memory_variables)
    #     | itemgetter("history")
    # )
    # # standalone = {
    # #     "standalone_question":{
    # #         "question":lambda x: x["question"],
    # #         "chat_history": lambda x: get_buffer_string(x["chat_history"])
    # #     }
    # #     | STANDALONE
    # #     | llm
    # #     | (lambda x:x.content if hasattr(x, "content") else x)
    # # }

    # def standalone(question_and_memory:dict):
    #     """return the standalone question

    #     Args:
    #         question_and_memory (dict): "question" key is the user input while \
    #             "chat_history" key is the chat_history

    #     Returns:
    #         dict: only one key named "standalone_question" with the value of the standalone question str
    #     """
    #     chat_history = get_buffer_string(question_and_memory["chat_history"])
    #     if debug:
    #         with open("_debug.txt", "w", encoding="utf-8") as f:
    #             # json.dump({"chat_history":chat_history}, f, indent=4)
    #             f.write(f"chat_history: {chat_history}\n\n")
    #     if chat_history != "":
    #         # has chat_history, need to LLM to generate the final question
    #         if debug:
    #             print("## DEBUG: has history, generate the standalone question by LLM")
    #         question_info = {
    #             "question":question_and_memory["question"],
    #             "chat_history": chat_history
    #         }
    #         standalone_prompt = STANDALONE.invoke(question_info)
    #         llm_return = llm.invoke(standalone_prompt)
    #         return {
    #             "standalone_question":llm_return.content if hasattr(llm_return, "content") else llm_return
    #         }
    #     else:
    #         # no history, the standalone question is the user input
    #         return {
    #             "standalone_question": question_and_memory["question"]
    #         }
    # 2. retrieve documents
    def measure_retrieval_cost(input_dict):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # docs = retriever.invoke(input_dict["standalone_question"])
        docs = retriever.invoke(input_dict["question"])
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        retrieval_time = (end_time - start_time) * 1000  # to ms
        retrieval_mem = (end_memory - start_memory)
        if debug:
            # print time cost for debug
            print(f"## DEBUG: Query time: {retrieval_time:.2f} ms (returned {len(docs)} documents)")
            print(f"## DEBUG: Query mem cost: {retrieval_mem:.2f} MB")
        return docs

    # retriever = db.as_retriever(search_kwargs={"k":20})
    # just think about the top 5 chunks for efficiency as well as reducing the context for LLM
    retriever = db.as_retriever(search_kwargs={"k":5})
    # retrieved_documents = {
    #     # "docs": itemgetter("standalone_question") | retriever,
    #     "docs": measure_retrieval_cost,
    #     # "question": lambda x: x["standalone_question"]
    #     "question": lambda x: x["question"]
    # }
    
    # because retrieved_documents becomes the first component in final_chain, it should be a Runnable
    retrieved_documents = RunnablePassthrough.assign(
        docs=lambda x: measure_retrieval_cost(x),
        question=lambda x: x["question"]
    )

    # 3. final input
    final_inputs = {
        "context": lambda x:_combine_documents(x["docs"]),
        "question": itemgetter("question")
    }

    def question_str(question:dict):
        """transform the doc from document to str"""
        context = _combine_documents(question["docs"])
        standalone_question = question["question"]
        final_question = {
            "context": context,
            "question": standalone_question
        }
        if debug:
            # record the final question
            with open("_debug.txt",'a', encoding="utf-8") as f:
                f.write(f"context: {final_question['context']}\n\n")
                f.write(f"question: {final_question['question']}\n\n")
            print("## DEBUG: the question is recorded in _debug.txt")
        return final_question
    
    # 4. answer
    # Create metrics dictionary for this chat chain
    metrics = {}

    answer = {
        # "answer": final_inputs
        "answer": question_str
        | FINAL_QUESTION
        | (lambda x: measure_llm_time(llm, x, local_model, debug, metrics)),
        "docs": itemgetter("docs")
    }

    # final_chain = loaded_memory | standalone | retrieved_documents | answer
    final_chain = retrieved_documents | answer
    # 5. chat
    def chat(question:str):
        # if question.lower() == "/clear":
        #     clear_conversation_history()
        # else:
        # Clear metrics for this chat session
        metrics.clear()
        inputs = {"question":question}
        total_start_time = time.time()
        # invoke
        result = final_chain.invoke(inputs)
        total_end_time = time.time()
        total_time = (total_end_time - total_start_time) * 1000  # to ms
        if debug:
            # print total time cost
            print(f"## DEBUG: Total time: {total_time:.2f} ms")
        # # store memory
        # memory.save_context(inputs, {"answer": result["answer"].content if hasattr(result["answer"], "content") else result["answer"]})

    return chat

def _combine_documents(docs: list, String_format=DOCUMENT_TO_STR, \
    separator="\n\n", return_str:bool = True)->str|list:
    """combine a list of documents to a string

    Args:
        docs (list): list of documents
        String_format: format template for documents
        separator (str): separator between documents
        return_str (bool): whether return str or the list

    Returns:
        str: string contains all documents (will input to LLM as the context)
        list: the doc list for debug (if retrun_str set to False)
    """
    # method1: just combine the context into a string
    doc_strings = [format_document(doc, String_format) for doc in docs]
    if return_str:
        # return str
        return separator.join(doc_strings)
    else:
        # return list for debug
        return doc_strings

    # method2: add the statements that illustrate the priority of docs
    # experiments show that it is useless.
    # doc_strings = []
    # for i, doc in enumerate(docs):
    #     # Lower index means higher priority (higher similarity)
    #     if i == 0:
    #         priority_text = "Highest Priority (Most Relevant)"
    #     elif i == 1:
    #         priority_text = "High Priority (Highly Relevant)"
    #     elif i == 2:
    #         priority_text = "Medium Priority (Moderately Relevant)"
    #     else:
    #         priority_text = f"Low Priority (Less Relevant, Rank {i+1})"

    #     formatted_doc = format_document(doc, String_format)
    #     # Add clear priority information before the document
    #     doc_with_priority = f"[{priority_text}]\n{formatted_doc}"
    #     doc_strings.append(doc_with_priority)
    # if return_str:
    #     # return str
    #     return separator.join(doc_strings)
    # else:
    #     # return list for debug
    #     return doc_strings

def measure_llm_time(llm, input_dict, local_model: bool, debug: bool, metrics_dict=None):
    """
    Measure LLM response time and invoke the model
    
    Args:
        llm: LLM model object
        input_dict: Input dictionary for the LLM
        local_model: Whether using local model
        debug: Whether to enable debug mode
        metrics_dict: Dictionary to store performance metrics

    Returns:
        Response from the LLM
    """
    start_time = time.time()
    # LLM reason
    if local_model:
    # local model, use Streaming to output the answer
        response = llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]).invoke(input_dict)
    else:
        # for API model, just invoke because this model may not support output streaming by StreamingStdOutCallbackHandler()
        response = llm.invoke(input_dict)
        # API model, need to output the answer
        final_answer = response
        if hasattr(final_answer, "content"):
            print(f"{final_answer.content}")
        else:
            print(f"{final_answer}")
    end_time = time.time()
    llm_time = (end_time - start_time) * 1000  # to ms
    if debug:
        # print time cost for debug
        print(f"\n## DEBUG: LLM response time: {llm_time:.2f} ms")
    
    # Store performance metrics in the provided dictionary
    if metrics_dict is not None:
        metrics_dict['llm_time'] = llm_time
    
    return response



def process_single_question(llm, retriever, question: str, debug: bool, local_model: bool):
    """
    Process single question independently (for batch testing)
    Uses the same complete chain as getChatChain from retrieval to answer generation
    
    Args:
        llm: LLM model object
        retriever: Retriever object
        question: Question text
        debug: Whether to enable debug mode
        local_model: Whether using local model
        
    Returns:
        dict: Dictionary containing answer, context and performance metrics
    """
    try:
        # Create metrics dictionary to store performance metrics
        metrics = {}
        
        # Create the complete chain (same as getChatChain)
        # 1. Measure retrieval cost (same as getChatChain)
        def measure_retrieval_cost(input_dict):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            
            docs = retriever.invoke(input_dict["question"])
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            retrieval_time = (end_time - start_time) * 1000  # to ms
            retrieval_mem = (end_memory - start_memory)
            if debug:
                print(f"## DEBUG: Query time: {retrieval_time:.2f} ms (returned {len(docs)} documents)")
                print(f"## DEBUG: Query mem cost: {retrieval_mem:.2f} MB")
            
            # Store performance metrics in the metrics dictionary
            metrics['retrieval_time'] = retrieval_time
            metrics['retrieval_mem'] = retrieval_mem
            return docs
        
        # 2. Build question_str function (same as getChatChain)
        def question_str(question_dict):
            """transform the doc from document to str"""
            context = _combine_documents(question_dict["docs"])
            standalone_question = question_dict["question"]
            final_question = {
                "context": context,
                "question": standalone_question
            }
            return final_question

        # 3. Build retrieved_documents (same as getChatChain)
        # use RunnablePassthrough to make it a Runnable for chain (the first component should be a Runnable)
        retrieved_documents = RunnablePassthrough.assign(
            docs=lambda x: measure_retrieval_cost(x), # a list of document objects
            question=lambda x: x["question"]
        )

        # 4. Build answer (same as getChatChain)
        answer = {
            "answer": question_str | FINAL_QUESTION | (lambda x: measure_llm_time(llm, x, local_model, debug, metrics)),
            "docs": itemgetter("docs") # a list of document objects
        }

        # 5. Create the final chain (same as getChatChain)
        final_chain = retrieved_documents | answer
        
        # 6. Invoke the complete chain (same as getChatChain)
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        
        # 7. Extract results
        response = result["answer"]
        docs = result["docs"]
        answer_content = response.content if hasattr(response, "content") else str(response)
        answer_content = remove_thinking_process(answer_content)
        
        # Extract performance metrics from the metrics dictionary
        retrieval_time = metrics.get('retrieval_time', 0)
        retrieval_mem = metrics.get('retrieval_mem', 0)
        llm_time = metrics.get('llm_time', 0)

        # Build context for return
        context_str = _combine_documents(docs)
        context_list = [format_document(doc, DOCUMENT_TO_STR) for doc in docs] # just return the doc without priority because priority is not in DB
        # context_list = _combine_documents(docs, return_str=False) # return list
        retrieval_time_str = f"{retrieval_time:.2f} ms"
        retrieval_mem_str = f"{retrieval_mem:.2f} MB"
        llm_time_str = f"{llm_time:.2f} ms"
        return {
            "answer": answer_content,
            "context_str": context_str,
            "context_list": context_list,
            "docs_count": len(docs),
            "retrieval_time": retrieval_time_str,
            "retrieval_mem": retrieval_mem_str,
            "llm_time": llm_time_str
        }

    except Exception as e:
        print(f"Error processing question: {e}")
        return None

def remove_thinking_process(answer:str):
    """remove the thinking process in the answer

    Args:
        answer (str): the answer may contain thinking process
    """
    if "<think>" in answer and "</think>" in answer:
        new_answer = re.sub(r'<think>.*?</think>','',answer, flags=re.DOTALL)
        return new_answer
    else:
        return answer
# def clear_conversation_history():
#     global _memory_instance
#     if _memory_instance is not None:
#         _memory_instance.clear()
#         print("Conversation history cleared.")