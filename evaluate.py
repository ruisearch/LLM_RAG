"""
Evaluate the outputs of RAG
"""
import os
import json
from pydantic import SecretStr
import time
import argparse
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness


def main(response_path:str, context_path:str):
    print(f"response: {response_path}")
    print(f"context: {context_path}")
    query_path = os.path.join('openragbench','queries_sample_100.json')
    answer_path = os.path.join('openragbench', 'answers_sample_100.json')

    # Load all necessary data files
    print("Loading data files...")
    with open(query_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    with open(answer_path, 'r', encoding='utf-8') as f:
        answers = json.load(f)

    with open(response_path, 'r', encoding='utf-8') as f:
        responses = json.load(f)

    with open(context_path, 'r', encoding='utf-8') as f:
        contexts = json.load(f)

    # construct the sample_queries and expected_answers
    sample_queries = []
    sample_answers = []
    responses_list = []
    context_list = []

    for id,data in queries.items():
        sample_queries.append(data["query"])
        sample_answers.append(answers[id])
        responses_list.append(responses[id]["response"])
        context_list.append(contexts[id]["context"])

    dataset = []
    for query,reference,Response,Context in \
        zip(sample_queries,sample_answers,responses_list,context_list):
        dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": Context,
                "response": Response,
                "reference": reference
            }
        )

    # # Save datasets to file
    # with open('test.txt', 'w', encoding='utf-8') as f:
    #     f.write("=== QWEN DATASET ===\n")
    #     f.write(json.dumps(dataset, ensure_ascii=False, indent=2))
    evaluate_dataset = EvaluationDataset.from_list(dataset)

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

    print(f"\nEvaluating rag with {len(dataset)} samples...")
    try:
        start_time = time.time()
        result = evaluate(dataset=evaluate_dataset,metrics=\
            [LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
        
        # just compute recall
        # result = evaluate(dataset=evaluate_dataset,metrics=\
        #     [FactualCorrectness(mode='recall')],llm=evaluator_llm)
        
        # just compute precision
        # result = evaluate(dataset=evaluate_dataset,metrics=\
        #     [FactualCorrectness(mode='precision')],llm=evaluator_llm)
        end_time = time.time()
        cost = end_time - start_time
        print(f"rag evaluation costs {cost:.2f} s")
        print(f"rag evaluation result: {result}")
    except Exception as e:
        print(f"Error evaluating rag: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate the response of LLM by Ragas.')
    parser.add_argument(
        "-r",
        "--responsePath",
        help="Path to the response json"
    )
    parser.add_argument(
        "-c",
        "--contextPath",
        help="Path to the context path"
    )
    args = parser.parse_args()
    main(args.responsePath, args.contextPath)