"""
Simplified API configuration management module
Used to manage API Key configuration, supports Windows and Linux environments
"""

import os
from typing import Optional


class APIConfigManager:
    """Simplified API configuration manager"""

    # Supported API service provider configurations
    # support openai, ds and qwen models currently
    API_PROVIDERS = {
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "env_var": "DASHSCOPE_API_KEY"
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "env_var": "OPENAI_API_KEY"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "env_var": "DEEPSEEK_API_KEY"
        }
    }
    
    def _infer_provider(self, model_name: str) -> Optional[str]:
        """
        Infer service provider based on model name
        
        Args:
            model_name: Model name
            
        Returns:
            Service provider name, returns None if cannot infer
        """
        model_lower = model_name.lower()
        
        # Qwen model recognition
        if model_lower.startswith("qwen"):
            return "qwen"
        
        # DeepSeek model recognition
        if model_lower.startswith("deepseek"):
            return "deepseek"
        
        # OpenAI model recognition (default)
        if any(keyword in model_lower for keyword in ["gpt", "openai"]):
            return "openai"
        
        # Default return openai
        return "openai"
    
    def get_api_config(self, model_name: str) -> Optional[dict]:
        """
        Get API configuration based on model name
        
        Args:
            model_name: Model name
            
        Returns:
            Dictionary containing api_key and base_url, returns None if not found
        """
        # Infer service provider based on model name
        provider = self._infer_provider(model_name)
        if not provider or provider not in self.API_PROVIDERS:
            return None
        
        # Get API Key
        env_var = self.API_PROVIDERS[provider]["env_var"]
        api_key = os.environ.get(env_var)
        
        # if api_key:
        #     print(f"✅ Retrieved API Key from environment variable {env_var}")
        # else:
        #     print(f"⚠️  Environment variable {env_var} not set")
        #     return None
        if not api_key:
            # can not get the API key
            return None
        
        # Return configuration information
        return {
            "api_key": api_key,
            "base_url": self.API_PROVIDERS[provider]["base_url"],
            "provider": provider
        }
    
    # def print_setup_instructions(self, model_name: str) -> None:
    #     """
    #     print how to set API key
    #     
    #     Args:
    #         model_name: name of model
    #     """
    #     provider = self._infer_provider(model_name)
    #     if not provider or provider not in self.API_PROVIDERS:
    #         print(f"❌ Cannot identify provider for model {model_name}")
    #         return
    #     
    #     env_var = self.API_PROVIDERS[provider]["env_var"]
    #     
    #     print(f"\n📋 {provider.upper()} API Key Setup Instructions:")
    #     print("=" * 50)
    #     
    #     # Windows PowerShell
    #     print("Windows PowerShell:")
    #     print(f"   $env:{env_var} = \"your_api_key_here\"")
    #     print(f"   # Or set permanently (requires terminal restart):")
    #     print(f"   [Environment]::SetEnvironmentVariable(\"{env_var}\", \"your_api_key_here\", \"User\")")
    #     
    #     # Windows CMD
    #     print("\nWindows CMD:")
    #     print(f"   set {env_var}=your_api_key_here")
    #     
    #     # Linux/Mac
    #     print("\nLinux/Mac:")
    #     print(f"   export {env_var}=your_api_key_here")
    #     print(f"   # Or add to ~/.bashrc or ~/.zshrc:")
    #     print(f"   echo 'export {env_var}=your_api_key_here' >> ~/.bashrc")
    #     
    #     print(f"\n Get API Key:")
    #     if provider == "qwen":
    #         print("   Visit: https://dashscope.aliyuncs.com/")
    #     elif provider == "openai":
    #         print("   Visit: https://platform.openai.com/api-keys")
    #     elif provider == "deepseek":
    #         print("   Visit: https://platform.deepseek.com/")
    #     
    #     print("=" * 50)


# Convenience functions for API configuration management
def get_api_config(model_name: str) -> Optional[dict]:
    """
    Get API configuration for a model (convenience function)
    
    Args:
        model_name: Model name
        
    Returns:
        Dictionary containing api_key and base_url, returns None if not found
    """
    manager = APIConfigManager()
    return manager.get_api_config(model_name)


def infer_provider(model_name: str) -> Optional[str]:
    """
    Infer service provider based on model name (convenience function)
    
    Args:
        model_name: Model name
        
    Returns:
        Service provider name, returns None if cannot infer
    """
    manager = APIConfigManager()
    return manager._infer_provider(model_name)


