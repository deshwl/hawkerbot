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
    st.session_state.messages = [{"role": "assistant", "content": "你好，烹饪大师！问我有关小贩摊位投标的问题!"}]

st.set_page_config(page_title="与小贩机器人聊天 🤖💬", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.sidebar.success("选择上面的语言")
with st.sidebar.expander("过去的聊天记录", expanded=True):
    st.markdown("""
    `[ 22 Jan 2024 ]`
    你: 非常感谢你的帮助! 👍  
    """)
st.sidebar.button('Clear Screen', on_click=clear_screen)
st.image("./images/logo.png")
st.title("与小贩机器人聊天 🤖💬")
st.markdown('''
<div>
<div style="border: 0.3px solid gray; padding: 10px; border-radius: 10px; margin: 10px 0px;">
<p><b>小贩摊位投标</b><br>
<i>示例问题：忠忠美食中心熟食摊位的最高出价是多少?</i></p>
</div>
<div style="border: 0.3px solid gray; padding: 10px; border-radius: 10px; margin: 10px 0px;">
<p><b>根据预算推荐小贩摊位</b><br>
<i>示例问题：如果我的预算是$500，我可以标到哪一个小贩中心的摊位?</i></p>
</div>
</div>
''', unsafe_allow_html=True)

# Setup Bedrock
region='us-east-1'

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
    aws_access_key_id=st.secrets["AWS_ACCESS_ID"],
    aws_secret_access_key=st.secrets["AWS_ACCESS_KEY"]
)

llm = Bedrock(client=bedrock_runtime, model = "anthropic.claude-3-5-sonnet-20240620-v1:0", system_prompt="You are an AI assistant and your job is to answer questions about the data you have. Keep your answers short, concise and do not hallucinate. If the user ask questions that you don't know, apologize and say that you cannot answer.")
embed_model = BedrockEmbedding(client=bedrock_runtime, model = "amazon.titan-embed-text-v1")

Settings.llm = llm
Settings.embed_model = embed_model

@st.cache_resource(show_spinner=False)
def load_data():
  with st.spinner(
    text="启动人工智能引擎。可能还要等一下..."):
    # load the documents and create the index
    documents = SimpleDirectoryReader(input_dir="data", recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

# Create Index
index=load_data()

# Initialize the chat messages history        
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，烹饪大师！问我有关小贩摊位投标的问题!"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys(): 
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and save to chat history
if prompt := st.chat_input("问我问题"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"].replace("$", "\$"))

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("请给我一分钟，让我想想..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response.replace("$", "\$"))
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
