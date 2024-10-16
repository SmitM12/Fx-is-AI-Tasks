from huggingface_hub import HfApi
import os

# Replace with your actual token
api_token = os.getenv('HUGGINGFACE_API_TOKEN')

# Initialize the Hugging Face API
api = HfApi()

try:
    # List models to check if the token is valid
    models = api.list_models()
    print("Token is valid! Available models:")
    for model in models:
        if model == "mistralai/Mistral-7B-v0.1":
            print("Valid: ",model)
except Exception as e:
    print(f"Error with token: {e}")
