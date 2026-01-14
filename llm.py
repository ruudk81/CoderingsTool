import os
import logging
from typing import Optional, Dict, Any, List
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureOpenAIClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        deployment_name_embedding: Optional[str] = None,
        use_azure_ad: bool = True  # Default to Azure AD auth
    ):

        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.deployment_name_embedding = deployment_name_embedding or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING")

        # Validate required parameters
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Set AZURE_OPENAI_ENDPOINT environment variable.")
        if not self.deployment_name:
            raise ValueError("Azure OpenAI deployment name is required. Set AZURE_OPENAI_DEPLOYMENT_NAME environment variable.")

        # Initialize the client
        try:
            if use_azure_ad:
                # Use Azure AD / Entra authentication (no API key needed)
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )
                self.client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=self.api_version
                )
                logger.info("Azure OpenAI client initialized with Azure AD authentication")
            else:
                # Fall back to API key authentication
                api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("API key required when not using Azure AD auth")
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.api_version
                )
                logger.info("Azure OpenAI client initialized with API key authentication")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:

        try:
            response = self.client.chat.completions.create(
                model=model or self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    def simple_chat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Simple chat failed: {e}")
            raise
    
    def get_embeddings(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:

        try:
            response = self.client.embeddings.create(
                model=model or self.deployment_name_embedding or self.deployment_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise


# Global client instance 
_llm_client = None

def get_llm_client(**kwargs) -> AzureOpenAIClient:

    global _llm_client
    if _llm_client is None:
        _llm_client = AzureOpenAIClient(**kwargs)
    return _llm_client

def reset_llm_client():
    global _llm_client
    _llm_client = None

