import streamlit as st
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage


# Streamlit 페이지 설정
st.title("팀스파르타 온보딩 안내봇🐥 ")

# 메시지 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4o")

pdf_paths = ["./teamsparta/vacation.pdf", "./teamsparta/Codebook.pdf",
             "./teamsparta/one-goal.pdf", "./teamsparta/Tuition Assistance.pdf",
             "./teamsparta/routine.pdf", "./teamsparta/Counseling.pdf",
             "./teamsparta/communication_culture.pdf", "./teamsparta/values.pdf",
             "./teamsparta/monthly_goal.pdf",
             ]  # 사용할 PDF 파일 경로 추가
docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 벡터스토어 생성
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory=".rtan_chatbot_chroma_data"
)
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

# 이전 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력을 기다리는 입력 창
if prompt := st.chat_input("질문을 입력해주세요."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 검색 및 답변 생성
    with st.spinner("검색 중..."):
        retrieved_docs = retriever.invoke(prompt)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # ChatPromptTemplate 생성
        prompt_template = ChatPromptTemplate.from_messages([
            HumanMessage(content="너는 팀스파르타라는 회사의 온보딩(신규 입사자들에게 회사 문화나 규정을 알려주는 것)을 도와주
는 챗봇이야. 첨부한 문서들을 바탕으로 팀스파르타에 대해 사용자가 묻는 질문에 답변해줘."
                                 "팀스파르타 문화와 관련 없는 질문이거나 잘 모르는 정보라면 답하지 말아줘. 다만 답할 수 있는 >내용에 대해서는 가능한 가장 자세하게 답변해줘.\n"
                                 f"Question: {prompt}\nContext: {format_docs(retrieved_docs)}")
        ])

        # 사용자 지시사항에 맞춰 프롬프트 생성
        user_prompt = prompt_template.format()


        # 모델 응답 생성
        response = llm.invoke(user_prompt)



    # 답변 출력 및 세션에 저장
    with st.chat_message("assistant"):
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
