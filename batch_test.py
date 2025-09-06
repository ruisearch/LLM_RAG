"""
Batch Testing Module
Handles batch testing functionality for RAG system
"""
import json
from tqdm import tqdm
import time
import random
from llm import process_single_question


def batch_test(llm, db, questions_file: str, output_file: str, \
    context_file: str, debug: bool, local_model: bool) -> None:
    """
    Batch testing function: Independent batch processing logic, not dependent on chat function
    
    Args:
        llm: LLM model object
        db: Vector database object
        questions_file: Path to questions file (JSON format)
        output_file: Path to output JSON file
        context_file: Path to context JSON file
        debug: Whether to enable debug mode
        local_model: Whether using local model
    """
    # print(f"Starting batch testing mode")
    # print(f"Reading questions file: {questions_file}")
    # print(f"Output file: {output_file}")
    # print(f"Context file: {context_file}")
    
    # Read questions JSON file
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        print(f"Successfully read {len(questions_data)} questions")
    except FileNotFoundError:
        print(f"Questions file not found: {questions_file}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON file format error: {e}")
        return
    
    # Initialize result dictionaries
    results = {}
    contexts = {}
    
    # Setup retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Use tqdm to display progress bar
    with tqdm(total=len(questions_data), desc="Processing QA", unit="items") as pbar:
        for qa_id, qa_data in questions_data.items():
            # Retry mechanism for API models
            max_retries = 3
            base_delay = 1.0 # 1 second
            result = None
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                # Process single question (already has try-catch inside)
                result = process_single_question(
                    llm, retriever, qa_data["query"], debug, local_model
                )
                
                if result:
                    break  # Success, exit retry loop
                else:
                    # Failed, check if we can retry
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1) # 1 second + random delay
                        print(f"\n##### DEBUG: Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s delay...")
                        time.sleep(delay)
                    else:
                        # All retries exhausted
                        print(f"qa_id {qa_id} failed, all {max_retries + 1} attempts exhausted")
                        break
            
            if result:
                # Save results to results dictionary
                results[qa_id] = {
                    "response": result["answer"],
                    "retrieval_time": result["retrieval_time"],
                    "retrieval_mem": result["retrieval_mem"],
                    "llm_time": result["llm_time"]
                }
                
                # Save context to contexts dictionary
                contexts[qa_id] = {
                    "context": result["context_list"]
                }
                
                pbar.set_postfix({"Status": "Success", "QAID": qa_id[:8]})
            else:
                pbar.set_postfix({"Status": "Failed", "QAID": qa_id[:8]})
            
            # Save JSON files after each question
            save_json_files(results, contexts, output_file, context_file)
            
            pbar.update(1)
    
    # Final save and summary
    print(f"Results saved to: {output_file}")
    print(f"Context saved to: {context_file}")
    
    print(f"Batch testing completed! Processed {len(results)} questions")



def save_json_files(results: dict, contexts: dict, output_file: str, context_file: str) -> None:
    """
    Save results and contexts to JSON files
    
    Args:
        results: Results dictionary to save
        contexts: Contexts dictionary to save
        output_file: Path to output JSON file
        context_file: Path to context JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving JSON files: {e}")