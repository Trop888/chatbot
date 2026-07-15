import streamlit as st
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

st.title("学习助手")
with st.sidebar:
    subject=st.selectbox(
        "选择学科领域",
        options=["文学 ","数学","计算机","英语"]
    )

    style=st.selectbox(
        "讲解风格",
        options=["简洁","详细"],
    )

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"你好，我是你的学习助手！"}]
    st.session_state["memory"]=ConversationBufferMemory(
        memory_key="chat_history",return_messages=True
    )


if "last_subject" not in st.session_state:
    st.session_state["last_subject"]=subject  

if subject!=st.session_state["last_subject"]:
    st.session_state["memory"].clear()
    st.session_state["messages"]=[{"role":"assistant","content":"你好，我是你的学习助手！"}]
    st.session_state["last_subject"]=subject
    st.rerun()

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])


def get_prompt_template(subject,style):
    style_dict={
        "简洁": "仅提供直接答案和最少的必要解释。不要添加额外细节、发散讨论或无关信息。保持回答清晰、简洁，目标是为用户快速提供解决方案。",
        "详细": "第一，针对用户提问给出直接答案和清晰的解释；第二，基于此提供必要的相关知识点的信息，以补充背景或加深理解。"
    }

    system_template = """你是{subject}领域的专家，只能回答{subject}相关的问题。
【重要规则】
1. 如果用户的问题与{subject}无关，你必须礼貌拒绝，例如："抱歉，我是{subject}领域的学习助手，无法回答其他领域的问题。请问您有{subject}方面的问题吗？"
2. 绝对不要回答与{subject}无关的问题，即使用户坚持要求。
3. 你需要遵循以下讲解风格：{style}。"""

    prompt_template=ChatPromptTemplate(
        [
            ("system",system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user","{input}")
        ],
        partial_variables={"subject":subject,"style":style_dict[style]},
    )
    return prompt_template

def generate_response(user_input,subject,style,memory):
    client=ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        temperature=0.0
    )
    prompt=get_prompt_template(subject,style)
    chain=ConversationChain(llm=client,memory=memory,prompt=prompt)
    response=chain.invoke({"input":user_input})
    return response["response"]

user_input=st.chat_input("你的问题/学习需求")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append({"role":"user","content":user_input})
    with st.spinner("AI正在思考中，请稍等..."):
        response=generate_response(
            user_input,subject,style,st.session_state["memory"]
        )
    st.chat_message("assistant").write(response)
    st.session_state["messages"].append({"role":"assistant","content":response})
