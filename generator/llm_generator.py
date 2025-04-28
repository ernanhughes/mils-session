# generator/llm_generator.py
# Unified interface for generating embeddings (and future generations) via LLMs

import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API key and model from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

def generate_embedding(text):
    """
    Generate a vector embedding from a given text prompt.

    Args:
        text (str): The input text to embed.

    Returns:
        list: A list of floats representing the embedding vector.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# (Future)
# def generate_image(prompt):
#     """
#     (Placeholder for future) Generate an image from a text prompt.
#     """
#     pass

if __name__ == "__main__":
    # Example usage
    test_text = "A small cabin in the forest by a lake"
    embedding = generate_embedding(test_text)
    if embedding:
        print(f"Generated embedding (length {len(embedding)} dimensions):")
        print(embedding)
