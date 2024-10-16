import streamlit as st
from langchain import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot with HuggingFace"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, please respond to the user queries. Note do not repeat system and user prompts in your response."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, llm_input, temperature, max_tokens):
    # openai.api_key = api_key
    print(f"API Key: {api_key}")
    print(f"Selected Model: {llm_input}")
    print(f"Temperature: {temperature}")
    print(f"Max Tokens: {max_tokens}")
    # os.environ['HUGGINGFACE_API_TOKEN'] = api_key
    
    # Initialize the LangChain LLM with Hugging Face model
    llm = HuggingFaceHub(
        repo_id=llm_input,
        model_kwargs= {"temperature": temperature, "max_length": max_tokens},
        huggingfacehub_api_token=api_key
    )
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot with HuggingFace")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your HugginFace Access Token:", type="password")

## Drop down to select various Hugging Face Models
llm_input = st.sidebar.selectbox("Select a Hugging Face Model", ["tiiuae/falcon-7b", "mistralai/Mistral-7B-v0.1", "tiiuae/falcon-7b-instruct", "EleutherAI/gpt-j-6B"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=512, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm_input, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")