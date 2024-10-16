import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Set Hugging Face API Token
hf_token = os.getenv('HUGGINGFACE_API_TOKEN')  # Use your actual token

# Create Hugging Face Endpoint
hf_endpoint = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    # model=
    task="text-generation",
    max_new_tokens=512,
    do_sample=True,  # Enable sampling to use temperature
    repetition_penalty=1.03,
    temperature=0.7,
    huggingfacehub_api_token=hf_token
)

# Initialize the ChatHuggingFace model
try:
    # Pass the Hugging Face endpoint directly to the model argument
    llm = ChatHuggingFace(llm=hf_endpoint)
    print("Model initialized successfully!")
except Exception as e:
    print(f"Error initializing model: {e}")
