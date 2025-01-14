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
    st.session_state.messages = [{"role": "assistant", "content": "你好，烹饪大师！问我有关小贩摊位投标的问题!"}]

# Function to set the question when a sample question is clicked
def set_question(question):
    st.session_state.question = question
    
st.set_page_config(page_title="小贩人工智能助手 🤖💬", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.sidebar.success("选择上面的语言")
with st.sidebar.expander("过去的聊天记录", expanded=True):
        st.markdown("""
        `[ 22 Dec 2024 ]`
        你: 非常感谢你的帮助! 👍  
        """)

col1, col2 = st.columns([3,1])
with col2:
    if st.button('清除聊天记录'):
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
    <div class="gradient-text">小贩人工智能助手 </div>
    """
st.markdown(gradient_text_html, unsafe_allow_html=True)

st.markdown("### 样题")

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
    if st.button("Maxwell 熟食中心 熟食摊位的最高出价是多少?"):
        set_question("Maxwell 熟食中心 熟食摊位的最高出价是多少?")
with col2:
    if st.button("我如何申请成为小贩？"):
        set_question("我如何申请成为小贩？")
with col3:
    if st.button("如果我的预算是$500，我可以标到哪一个小贩中心的摊位?"):
        set_question("如果我的预算是$500，我可以标到哪一个小贩中心的摊位?")

# Setup Bedrock
region='us-east-1'

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
    aws_access_key_id=st.secrets["AWS_ACCESS_ID"],
    aws_secret_access_key=st.secrets["AWS_ACCESS_KEY"]
)

llm = Bedrock(client=bedrock_runtime, model = "anthropic.claude-3-5-sonnet-20240620-v1:0", system_prompt="""你是一个人工智能助手，旨在帮助解决与小贩相关的查询。您的知识主要来自两个来源：

    1. 包含小贩摊位投标数据的 CSV 文件。这包括小贩中心名称、交易类型和出价金额等信息。
    2. 提供有关如何成为小贩的信息的 PDF 文档。

    回答问题时：
    1. 有关投标或投标金额的查询，请仅参考 CSV 文件。
    2. 有关成为小贩的流程、法规或有关小贩文化的一般信息的问题，请使用 PDF 文档中的信息。
    3. 让您的回答简洁、真实。
    4. 如果使用这些来源都无法回答问题，请说明“我当前的数据中没有该信息”。
    5. 请勿发明、假设或幻想这些文档中提供的信息之外的任何信息。
              
    根据预算提供建议时:
    1. 如果用户询问他们可以根据预算竞标哪个小贩中心（例如，“如果我的预算是$500，我可以竞标哪个小贩中心？”），请使用 CSV 文件查找合适的选项。

    您的职责是协助潜在和现有的小贩提供有关出价、成为小贩的流程的准确信息，并仅根据所提供的文件在需要时提供建议.""")

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
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，烹饪大师！问我有关小贩摊位投标的问题!"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state: 
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Always show the chat input
user_input = st.chat_input("问我问题")

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
        with st.spinner("请给我一分钟，让我想想..."):
            response = st.session_state.chat_engine.chat(new_input)
            st.markdown(response.response.replace("$", "\$"))

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.response})

            # Add feedback buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👍 有帮助"):
                    st.success("Thank you for your feedback!")
                    # Log the positive feedback here
            with col2:
                if st.button("👎 没有帮助"):
                    st.error("We're sorry to hear that. We'll work on improving.")
                    # Log the negative feedback here
            with col3:
                if st.button("🤔 不清楚"):
                    st.warning("We'll try to make our responses clearer.")
                    # Log the feedback about clarity here
