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

# Clear Chat History fuction
def clear_screen():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.messages = [{"role": "assistant", "content": "Greetings, Culinary Maestro! Ready to spice up your tender bids? Ask me anything about hawker stall tender bids!"}]
    st.session_state.first_time_user = False
# Function to show first-time user guide

def show_first_time_user_guide():
    st.markdown("## Welcome to HawkerBot! 🍜")
    st.markdown("Let's get you started with a quick guide:")
    st.markdown("### 1. Ask a Question")
    st.image("./images/question.JPG", width=600)
    st.markdown("Type your question about hawker stalls or becoming a hawker in the chat box at the bottom of the screen.")
    st.markdown("### 2. Get an Answer")
    st.image("./images/answer.JPG", width=600)
    st.markdown("HawkerBot will provide an answer based on the latest information about hawker stalls and regulations.")
    st.markdown("### 3. Explore More")
    st.markdown("Feel free to ask follow-up questions or explore different topics related to hawker stalls!")
    st.markdown("### Need Help?")
    st.markdown("If you're unsure what to ask, try these sample questions:")
    st.markdown("- What is the highest bid for a cooked food stall at Chomp Chomp Food Centre?")
    st.markdown("- How do I apply to become a hawker?")
    st.markdown("- What's the average rental price for a stall at Maxwell Food Centre?")
    if st.button("Got it! Let's start chatting"):
        st.session_state.first_time_user = False
        st.rerun()

st.set_page_config(page_title="HawkerBot 🤖💬", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.sidebar.success("Select language above")

# Initialize session state for first-time user

if "first_time_user" not in st.session_state:
    st.session_state.first_time_user = True

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
    1. For queries about tender bids, or bid amounts, refer exclusively to the CSV files.
    2. For questions about the process of becoming a hawker, regulations, or general information about hawker culture, use the information from the PDF documents.
    3. Keep your responses concise and factual.
    4. If a question can't be answered using either of these sources, state "I don't have that information in my current data."
    5. Do not invent, assume, or hallucinate any information beyond what's provided in these documents.
    Your role is to assist potential and current hawkers with accurate information about bids and the process of becoming a hawker, based solely on the provided documents.""")

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
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings, Culinary Maestro! Ready to spice up your tender bids? Ask me anything about hawker stall tender bids!"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys(): 
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Initialize session state for first-time user

# Main app logic

if st.session_state.first_time_user:
    show_first_time_user_guide()
else:
    with st.sidebar.expander("Past Chat", expanded=True):
        st.markdown("""
        `[ 22 Jan 2024 ]`
        You: Thank you so much for your help! 👍  
        """)

    # Add a way for users to revisit the guide

    if not st.session_state.first_time_user:
            col1, col2, col3 = st.columns([2,1,1])
            with col2:
                if st.button('Clear Chat History'):
                    clear_screen()
            with col3:
                if st.button("Show Guide Again"):
                    st.session_state.first_time_user = True
                    st.rerun()
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
    st.markdown('''
    <div>
    <div style="border: 0.3px solid gray; padding: 10px; border-radius: 10px; margin: 10px 0px;">
    <p><b>Bids for hawker stall</b><br>
    <i>Sample Question: What is the highest bid for Chomp Chomp Food Centre cooked food stall?</i></p>
    </div>
    <div style="border: 0.3px solid gray; padding: 10px; border-radius: 10px; margin: 10px 0px;">
    <p><b>Recommendation for hawker stall based on budget</b><br>
    <i>Sample Question: If my budget is $500, which Hawker Centre stall can I rent?</i></p>
    </div>
    </div>
    ''', unsafe_allow_html=True)

        
    # Prompt for user input and save to chat history
    if prompt := st.chat_input("Ask me a question"): 
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the prior chat messages
    for message in st.session_state.messages: 
        with st.chat_message(message["role"]):
            st.write(message["content"].replace("$", "\$"))

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("One minute, cooking up a storm..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response.replace("$", "\$"))
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

                # Add feedback buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Helpful"):
                        st.success("Thank you for your feedback!")
                        # Here you could log the positive feedback
                with col2:
                    if st.button("👎 Not Helpful"):
                        st.error("We're sorry to hear that. We'll work on improving.")
                        # Here you could log the negative feedback
                with col3:
                    if st.button("🤔 Unclear"):
                        st.warning("We'll try to make our responses clearer.")
                        # Here you could log the feedback about clarity
