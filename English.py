import streamlit as st
import boto3
from llama_index.core import ( 
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.settings import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding, Models

# Function to clear chat history
def clear_chat_history():
    if "messages" in st.session_state:
        del st.session_state.messages
    if "chat_engine" in st.session_state:
        del st.session_state.chat_engine

# Check if the page has changed
if "current_page" not in st.session_state:
    st.session_state.current_page = __file__
elif st.session_state.current_page != __file__:
    clear_chat_history()
    st.session_state.current_page = __file__

# Clear Chat History fuction
def clear_screen():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.messages = [{"role": "assistant", "content": "Greetings, Culinary Maestro! Ready to spice up your tender bids? Ask me anything about hawker stall tender bids!"}]

# Function to set the question when a sample question is clicked
def set_question(question):
    st.session_state.question = question

# Define custom CSS
custom_css = """
<style>
    [data-testid="stSidebar"] {
        background-color: #a8e6cf;  /* This is a light greenish-blue color */
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #a8e6cf;  /* This ensures the scrollable part of the sidebar also has the same color */
    }
    [data-testid="stSidebarNav"] {
        background-color: rgba(168, 230, 207, 0.1);  /* Slightly transparent version of the same color for the navigation */
    }
    [data-testid="stSidebarNav"]::before {
        background-color: #a8e6cf;  /* Color for the navigation header */
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

st.set_page_config(page_title="HawkerBot 🤖💬", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.sidebar.success("Select language above")
with st.sidebar.expander("Past Chat", expanded=True):
        st.markdown("""
        `[ 22 Dec 2024 ]`
        You: Thank you so much for your help! 👍  
        """)

col1, col2 = st.columns([3,1])
with col2:
    if st.button('Clear Chat History'):
        clear_screen()

st.image("./images/logo.png")
gradient_text_html = """
    <style>
    .gradient-text {
        font-weight: bold;
        background: -webkit-linear-gradient(left, green, lightblue);
        background: linear-gradient(to right, green, lightblue);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-size: 3em;
    }
    </style>
    <div class="gradient-text">HawkerBot </div>
    """
st.markdown(gradient_text_html, unsafe_allow_html=True)

st.markdown("### Sample Questions")

# Custom CSS for the buttons
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 75px;
        white-space: normal;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# Create three columns for the buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What is the highest bid for Chomp Chomp Food Centre cooked food stall?"):
        set_question("What is the highest bid for Chomp Chomp Food Centre cooked food stall?")
with col2:
    if st.button("How do I apply to become a hawker?"):
        set_question("How do I apply to become a hawker?")
with col3:
    if st.button("If my budget is $500, which Hawker Centre can I bid for halal food?"):
        set_question("If my budget is $500, which Hawker Centre can I bid for halal food?")

# Setup Bedrock
region='us-east-1'

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
    aws_access_key_id=st.secrets["AWS_ACCESS_ID"],
    aws_secret_access_key=st.secrets["AWS_ACCESS_KEY"]
)

llm = Bedrock(client=bedrock_runtime, model = "anthropic.claude-3-5-sonnet-20240620-v1:0", system_prompt="""You are an AI assistant designed to help with hawker-related queries. Your knowledge comes from two main sources:
 
    1. CSV file containing data on hawker stall tender bids. This includes information such as name of hawker centre, trade types and bid amounts.
    2. PDF document that provide information on how to become a hawker.

    When answering questions:
    1. For queries about tender bids, or bid amounts, refer exclusively to the CSV file.
    2. For questions about the process of becoming a hawker, regulations, or general information about hawker culture, use the information from the PDF document.
    3. Keep your responses concise and factual.
    4. If a question can't be answered using either of these sources, state "I don't have that information in my current data."
    5. Do not invent, assume, or hallucinate any information beyond what's provided in these documents.
              
    When providing recommendations based on budget:
    1. If a user asks about which hawker centre they can bid for based on their budget (e.g., "If my budget is $500, which Hawker Centre can I bid for?"), use the CSV file to find suitable options.

    Your role is to assist potential and current hawkers with accurate information about bids, the process of becoming a hawker, and provide recommendations when requested, based solely on the provided documents.""")

embed_model = BedrockEmbedding(client=bedrock_runtime, model = "amazon.titan-embed-text-v1")

Settings.llm = llm
Settings.embed_model = embed_model

@st.cache_resource(show_spinner=False)
def load_data():
  with st.spinner(
    text="Starting up AI engine. This may take a while..."):
    # load the documents and create the index
    documents = SimpleDirectoryReader(input_dir="data", recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

# Create Index
index=load_data()

# Initialize the chat messages history        
if "messages" not in st.session_state: 
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings, Culinary Maestro! Ready to spice up your tender bids? Ask me anything about hawker stall tender bids!"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Always show the chat input
user_input = st.chat_input("Ask me a question")

# Display the prior chat messages
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"].replace("$", "\$"))

# Process new input (either from sample question or user input)
new_input = None
if "question" in st.session_state:
    new_input = st.session_state.question
    del st.session_state.question
elif user_input:
    new_input = user_input

if new_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": new_input})
   
    # Display user message
    with st.chat_message("user"):
        st.markdown(new_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("One minute, cooking up a storm..."):
            response = st.session_state.chat_engine.chat(new_input)
            st.markdown(response.response.replace("$", "\$"))

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.response})

            # Add feedback buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👍 Helpful"):
                    st.success("Thank you for your feedback!")
                    # Log the positive feedback here
            with col2:
                if st.button("👎 Not Helpful"):
                    st.error("We're sorry to hear that. We'll work on improving.")
                    # Log the negative feedback here
            with col3:
                if st.button("🤔 Unclear"):
                    st.warning("We'll try to make our responses clearer.")
                    # Log the feedback about clarity here
