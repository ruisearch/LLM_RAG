"""
get models. 
For local models, pull them from Ollama; For API usage, check the API configuration
Return the LLM object as well
"""
import ollama
from tqdm import tqdm
from api_config import get_api_config
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def __pull_model(name: str) -> None:
    current_digest, bars = "", {}
    for progress in ollama.pull(name, stream=True):
        digest = progress.get("digest", "")
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()

        if not digest:
            print(progress.get("status"))
            continue

        if digest not in bars and (total := progress.get("total")):
            bars[digest] = tqdm(
                total=total, desc=f"pulling {digest[7:19]}", unit="B", unit_scale=True
            )

        if completed := progress.get("completed"):
            bars[digest].update(completed - bars[digest].n)

        current_digest = digest


def __is_model_available_locally(model_name: str) -> bool:
    try:
        ollama.show(model_name)
        # raise ollama.ResponseError if model is not available
        return True
    except ollama.ResponseError:
        return False


def get_list_of_models() -> list[str]:
    """
    Retrieves a list of available models from the Ollama repository.

    Returns:
        list[str]: A list of model names available in the Ollama repository.
    """
    return [model["model"] for model in ollama.list()["models"]]


def check_if_model_is_available(model_name: str, is_local: bool = True) -> None:
    """
    Check if the model is available (local model or API model)
    For local models: check if downloaded, if not try to pull from Ollama repository
    For API models: check if API Key is configured

    Args:
        model_name (str): The name of the model to check
        is_local (bool): Whether it's a local model, True=local model, False=API model

    Raises:
        Exception: If the model is not available or configuration has issues
    """
    if is_local:
        # local model
        try:
            available = __is_model_available_locally(model_name)
        except Exception:
            raise Exception("Unable to communicate with the Ollama service")

        if not available:
            try:
                __pull_model(model_name)
            except Exception:
                raise Exception(
                    f"Unable to find model '{model_name}', please check the name and try again."
                )
    else:
        # use model by API; check API key cofiguration
        # Get API configuration
        api_config = get_api_config(model_name)
        
        if api_config is None:
            # the model does have any API key accordingly
            print(f"API model {model_name} configuration failed")
            print(f"Please set the corresponding API Key environment variable")
            
            # Print setup instructions
            # api_config_manager.print_setup_instructions(model_name)
            
            raise Exception(
                f"API model '{model_name}' API Key not configured, please set the corresponding environment variable"
            )
        # else:
        #     print(f"✅ API model {model_name} configured successfully")
        #     print(f"   Provider: {api_config['provider']}")
        #     print(f"   API Base URL: {api_config['base_url']}")

def create_llm(model_name: str, is_local: bool = True):
    """
    Create LLM object (local model or API model)
    
    Args:
        model_name (str): Model name
        is_local (bool): Whether it's a local model, True=local model, False=API model
        
    Returns:
        LLM object
    """
    if is_local:
        # Create local model LLM object
        # ChatOllama is a wrapper class provided by LangChain for interacting with Ollama local large models
        return ChatOllama(model=model_name)
    else:
        # Create API model LLM object
        
        # Get API configuration
        api_config = get_api_config(model_name)
        
        if api_config is None:
            raise Exception(f"API model {model_name} API Key not configured")

        api_key_str = api_config['api_key']
        # print("key:",api_key_str)
        try:
            # Create LLM object using OpenAI compatible interface
            return ChatOpenAI(
                model=model_name,
                api_key=SecretStr(api_key_str) if api_key_str else None,
                base_url=api_config['base_url'],
                temperature=0.1, # set 0.1 for reproducibility
                streaming=True # enable for qwen3-30b-a3b
            )
        except ImportError:
            raise Exception("Please install langchain-openai: pip install langchain-openai")
