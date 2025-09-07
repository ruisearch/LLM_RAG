"""
Evaluate the outputs of RAG (based on qwen3-30b-a3b-instruct-2507 or gpt-oss-20b currently)
"""
import os
import json
from pydantic import SecretStr
import time
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

query_path = os.path.join('openragbench','queries_sample_100.json')
answer_path = os.path.join('openragbench', 'answers_sample_100.json')

# gpt_response_path = os.path.join('benchResult','gpt_oss_output.json')
# gpt_context_path = os.path.join('benchResult','gpt_oss_context.json')

# qwen_response_path = os.path.join('benchResult','qwen_output.json')
# qwen_context_path = os.path.join('benchResult','qwen_context.json')
qwen_response_path = os.path.join('benchResult','qwen3_30b_a3b_output.json')
qwen_context_path = os.path.join('benchResult','qwen3_30b_a3b_context.json')

# Load all necessary data files
print("Loading data files...")
with open(query_path, 'r', encoding='utf-8') as f:
    queries = json.load(f)

with open(answer_path, 'r', encoding='utf-8') as f:
    answers = json.load(f)

with open(qwen_response_path, 'r', encoding='utf-8') as f:
    qwen_responses = json.load(f)

# with open(gpt_response_path, 'r', encoding='utf-8') as f:
#     gpt_responses = json.load(f)

with open(qwen_context_path, 'r', encoding='utf-8') as f:
    qwen_contexts = json.load(f)

# with open(gpt_context_path, 'r', encoding='utf-8') as f:
#     gpt_contexts = json.load(f)

# construct the sample_queries and expected_answers
# construct the responses of qwen-based rag and gpt-based rag
sample_queries = []
sample_answers = []
qwen_responses_list = []  # Rename to avoid conflict with loaded dictionaries
# gpt_responses_list = []   # Rename to avoid conflict with loaded dictionaries
qwen_context_list = []    # Rename to avoid conflict with loaded dictionaries
# gpt_context_list = []     # Rename to avoid conflict with loaded dictionaries

for id,data in queries.items():
    sample_queries.append(data["query"])
    sample_answers.append(answers[id])
    qwen_responses_list.append(qwen_responses[id]["response"])  # Use renamed list
    # gpt_responses_list.append(gpt_responses[id]["response"])    # Use renamed list
    qwen_context_list.append(qwen_contexts[id]["context"])     # Use renamed list, note: contexts not context
    # gpt_context_list.append(gpt_contexts[id]["context"])       # Use renamed list, note: contexts not context



# Construct the datasets of qwen and gpt
qwen_dataset = []
# gpt_dataset = []
# for query,reference,qResponse,gResponse,qContext,gContext in \
#     zip(sample_queries,sample_answers,qwen_responses_list,gpt_responses_list,qwen_context_list,gpt_context_list):
#     qwen_dataset.append(
#         {
#             "user_input": query,
#             "retrieved_contexts": qContext,
#             "response": qResponse,
#             "reference": reference
#         }
#     )
#     gpt_dataset.append(
#         {
#             "user_input": query,
#             "retrieved_contexts": gContext,
#             "response": gResponse,
#             "reference": reference
#         }
#     )
for query,reference,qResponse,qContext in \
    zip(sample_queries,sample_answers,qwen_responses_list,qwen_context_list):
    qwen_dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": qContext,
            "response": qResponse,
            "reference": reference
        }
    )

# # Save datasets to file
# with open('test.txt', 'w', encoding='utf-8') as f:
#     f.write("=== QWEN DATASET ===\n")
#     f.write(json.dumps(qwen_dataset, ensure_ascii=False, indent=2))
#     f.write("\n\n=== GPT DATASET ===\n")
#     f.write(json.dumps(gpt_dataset, ensure_ascii=False, indent=2))
qwen_evaluate_dataset = EvaluationDataset.from_list(qwen_dataset)
# gpt_evaluate_dataset = EvaluationDataset.from_list(gpt_dataset)

# Construct the evaluator llm (gpt-4o)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
    
llm = ChatOpenAI(
    api_key=SecretStr(api_key) if api_key else None,
    base_url="https://sg.uiuiapi.com/v1", # uiuiapi
    model="gpt-4o" # the evaluation llm is gpt-4o
)
evaluator_llm = LangchainLLMWrapper(llm)

# first evaluate the qwen-based rag; reference the test_ragas.py.
print(f"\nEvaluating qwen-based rag with {len(qwen_dataset)} samples...")
try:
    start_time = time.time()
    qwen_result = evaluate(dataset=qwen_evaluate_dataset,metrics=\
        [LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
    end_time = time.time()
    cost = end_time - start_time
    print(f"qwen-based rag evaluation costs {cost:.2f} s")
    print(f"qwen-based rag evaluation result: {qwen_result}")
except Exception as e:
    print(f"Error evaluating qwen-based rag: {e}")
    import traceback
    traceback.print_exc()

# # second evaluate the gpt-based rag; reference the test_ragas.py.
# print(f"\nEvaluating gpt-based rag with {len(gpt_dataset)} samples...")
# try:
#     start_time = time.time()
#     gpt_result = evaluate(dataset=gpt_evaluate_dataset,metrics=\
#         [LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
#     end_time = time.time()
#     cost = end_time - start_time
#     print(f"gpt-based rag evaluation costs {cost:.2f} s")
#     print(f"gpt-based rag evaluation result: {gpt_result}")
# except Exception as e:
#     print(f"Error evaluating gpt-based rag: {e}")
#     import traceback
#     traceback.print_exc()

print("\n" + "="*50)
print("Evaluation completed!")
print("="*50)
