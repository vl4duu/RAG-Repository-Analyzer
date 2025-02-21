from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the CodeBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def embed_textual_metadata(content):
    """
       Embeds textual metadata using the OpenAI embedding model.
       """
    try:
        response = client.embeddings.create(model="text-embedding-ada-002",  # Updated OpenAI's text embedding model
        input=content)

        embedding = response.data[0].embedding
        return response.data[0].embedding

    except Exception as e:
        print(f"An error occurred while embedding text: {e}")
        raise



# # Embed a README or docstring
# textual_metadata = "This function calculates the sum of two numbers."
# metadata_embedding = embed_textual_metadata(textual_metadata)

# Function to generate embeddings for a code snippet
def generate_code_embedding(code_snippet):
    # Tokenize the code snippet
    inputs = tokenizer(
        code_snippet,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Take the mean of the embeddings across all tokens
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding_np = embedding.numpy().tolist()

    return embedding.numpy()
