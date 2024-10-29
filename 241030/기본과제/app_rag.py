import streamlit as st
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

# Streamlit 페이지 설정
st.title("🤖  ChatBot with RAG")

# 메시지 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# LLM 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# RAG 소스 URL 설정
source_url = "https://spartacodingclub.kr/blog/all-in-challenge_winner"

# 웹 문서 로더 초기화 (SoupStrainer로 원하는 내용만 로드)
loader = WebBaseLoader(
    web_paths=(source_url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("css-8lvslw")  # 필요한 HTML 클래스 이름으로 수정
        )
    ),
)

# 웹 문서 로드 및 텍스트 분할
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 벡터스토어 생성
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory=".chroma_data"  # 데이터 저장 경로 지정
)
retriever = vectorstore.as_retriever()

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

        # RAG 프롬프트 로드
        prompt_template = hub.pull("rlm/rag-prompt")
        user_prompt = prompt_template.invoke({"context": format_docs(retrieved_docs), "question": prompt})

        # 모델 응답 생성
        response = llm.invoke(user_prompt)

    # 답변 출력 및 세션에 저장
    with st.chat_message("assistant"):
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
