from llm import get_llm_client

def main():
    # Get the Azure OpenAI client (using API key auth)
    client = get_llm_client(use_azure_ad=False)
    
    print("=== Simple Chat Completion ===")
    try:
        # chat completion
        response = client.simple_chat(
            prompt="Wat is windenergie ?",
            system_message="Je bent een behulpzame assistent.",
            temperature=0.7,
            max_tokens=200
        )
        print("Chat Response:")
        print(response)
        
    except Exception as e:
        print(f"Chat completion failed: {e}")
    
    print("\n=== Getting Embeddings ===")
    try:
        # Get embeddings for text
        text = "energiebronnen zoals zonne en windenergie zijn duurzaam"
        embeddings = client.get_embeddings(text)
        
        print(f"Text: {text}")
        print(f"Embedding dimensions: {len(embeddings)}")
        
    except Exception as e:
        print(f"Embedding generation failed: {e}")

if __name__ == "__main__":
    main()