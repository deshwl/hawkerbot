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
    st.session_state.first_time_user = False
# Function to show first-time user guide

def show_first_time_user_guide():
    st.image("./images/logo.png")
    st.markdown("## 欢迎来到国家环境局小贩人工智能助手!")
    st.markdown("让我们通过快速指南开始您的旅程：")
    st.markdown("### 1. 问一个问题")
    st.image("./images/question.JPG", width=600)
    st.markdown("在屏幕底部的聊天框中输入有关小贩摊位投标或成为小贩的问题。")
    st.markdown("### 2. 得到答案")
    st.image("./images/answer.JPG", width=600)
    st.markdown("小贩人工智能助手将根据有关小贩摊位投标和法规的现有信息提供答案。")
    st.markdown("### 3. 探索更多")
    st.markdown("请随意提出后续问题或探索与小贩摊位相关的不同主题！")
    st.markdown("### 需要帮助吗？")
    st.markdown("如果您不确定要问什么，请尝试以下示例问题：")
    st.markdown("- Chomp Chomp 熟食中心的熟食摊位最高出价是多少？")
    st.markdown("- 如何投标小贩摊位？")
    st.markdown("- Maxwell 熟食中心的摊位平均租金是多少？")

    # Add the caveat here
    st.warning("""
    请注意:
    - 该小贩人工智能助手根据公开数据提供信息，可能无法反映实时变化.
    - 这些答复应仅用作一般指导，不能替代官方建议。
    - 如需最新、准确的信息，请务必参考政府官方消息来源或咨询相关机构。
    - 聊天机器人的知识仅限于其接受过训练的数据，可能无法涵盖与小贩相关的查询的所有方面。
    """)
    if st.button("知道了！我们开始聊天吧"):
        st.session_state.first_time_user = False
        st.rerun()
        
st.set_page_config(page_title="小贩人工智能助手 🤖💬", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.sidebar.success("选择上面的语言")

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

llm = Bedrock(client=bedrock_runtime, model = "anthropic.claude-3-5-sonnet-20240620-v1:0", system_prompt="""你是一个人工智能助手，旨在帮助解决与小贩相关的查询。您的知识主要来自两个来源：

    1. 包含小贩摊位投标数据的 CSV 文件。这包括小贩中心名称、交易类型和出价金额等信息。
    2. 提供有关如何成为小贩的信息的 PDF 文档。

    回答问题时：
    1. 有关投标或投标金额的查询，请仅参考 CSV 文件。
    2. 有关成为小贩的流程、法规或有关小贩文化的一般信息的问题，请使用 PDF 文档中的信息。
    3. 让您的回答简洁、真实。
    4. 如果使用这些来源都无法回答问题，请说明“我当前的数据中没有该信息”。
    5. 请勿发明、假设或幻想这些文档中提供的信息之外的任何信息。
              
    提供建议时:
    1. 如果用户要求根据其预算建议竞标哪个小贩中心，请使用 CSV 文件找到合适的选项。

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
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，烹饪大师！问我有关小贩摊位投标的问题!"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys(): 
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Initialize session state for first-time user

# Main app logic

if st.session_state.first_time_user:
    show_first_time_user_guide()
else:
    with st.sidebar.expander("过去的聊天记录", expanded=True):
        st.markdown("""
        `[ 22 Jan 2024 ]`
        你: 非常感谢你的帮助! 👍  
        """)

    # Add a way for users to revisit the guide

    if not st.session_state.first_time_user:
            col1, col2, col3 = st.columns([2,1,1])
            with col2:
                if st.button('清除聊天记录'):
                    clear_screen()
            with col3:
                if st.button("再次显示指南"):
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
    <div class="gradient-text">小贩人工智能助手 </div>
    """
    st.markdown(gradient_text_html, unsafe_allow_html=True)
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
