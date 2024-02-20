import streamlit as st
import boto3
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.langchain import LangChainLLM
from langchain.llms import Bedrock
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import BedrockEmbeddings

st.set_page_config(page_title="Chat with HawkerBOT 🤖💬", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.sidebar.success("Select language above")
with st.sidebar.expander("Past Chat", expanded=True):
    st.markdown("""
    `[ 22 Jan 2024 ]`
    You: Thank you so much for your help! 👍  
    """)
st.image("./images/logo.png")
st.title("Chat with HawkerBOT 🤖💬")
st.markdown('''
<div>
<div style="border: 0.3px solid gray; padding: 10px; border-radius: 10px; margin: 10px 0px;">
<p><b>Bids for Hawker stall</b><br>
<i>Sample Question: What is the highest bid for Chomp Chomp Food Centre cooked food stall?</i></p>
</div>
<div style="border: 0.3px solid gray; padding: 10px; border-radius: 10px; margin: 10px 0px;">
<p><b>Recommendation for Hawker stall based on budget</b><br>
<i>Sample Question: If my budget is $500, which Hawker Centre stall can I rent?</i></p>
</div>
</div>
''', unsafe_allow_html=True)
# Setup Bedrock
region='us-east-1'
new_session = boto3.Session(
aws_access_key_id=st.secrets["AWS_ACCESS_ID"],
aws_secret_access_key=st.secrets["AWS_ACCESS_KEY"])

bedrock_runtime = new_session.client(
    service_name='bedrock-runtime',
    region_name=region,
)

# LLM - Amazon Bedrock LLM using LangChain
model_id = "anthropic.claude-v2"
model_kwargs =  { 
    "max_tokens_to_sample": 4096,
    "temperature": 0.1,
    "top_k": 250,
    "top_p": 1,
}

llm = Bedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs
)

# Embedding Model - Amazon Titan Embeddings Model using LangChain
# create embeddings
bedrock_embedding = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v1",
)

# load in Bedrock embedding model from langchain
embed_model = LangchainEmbedding(bedrock_embedding)

# Service Context
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.system_prompt = "You are an AI assistant and your job is to answer questions about the data you have. Keep your answers short, concise and do not hallucinate. If the user ask questions that you don't know, apologize and say that you cannot answer."

# Initialize the chat messages history        
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, Human! Ask me a question related to Hawker stall tender bids!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
  with st.spinner(
    text="Starting up AI engine. This may take a while..."):
    reader=SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs=reader.load_data()
    index=VectorStoreIndex.from_documents(docs)
    return index

# Create Index
index=load_data()

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys(): 
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and save to chat history
if prompt := st.chat_input("Ask me a question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("One minute, cooking up a storm..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
