import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

# 감정 분류 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("jeonghyeon97/koBERT-Senti5", trust_remote_code=True)

# 감정 레이블과 이모지 매핑
emotion_labels = {
    0: ("Angry", "😡"),
    1: ("Fear", "😨"),
    2: ("Happy", "😊"),
    3: ("Tender", "🥰"),
    4: ("Sad", "😢")
}

# Streamlit 페이지 설정
st.title("감정 기반 추천 음악 챗봇 🎶")
st.write("일기나 오늘의 감정을 입력하면, 어울리는 음악을 추천해드립니다.")

# 메시지 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# PDF 문서 로더 초기화
pdf_path = "./playlist.pdf"  # 사용하려는 PDF 파일 경로
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 벡터스토어 생성
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory=".music_recommendation_chroma_data"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 이전 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
text = st.text_area("오늘의 일기를 입력하세요:")

if st.button("분석하기"):
    # 감정 분석 수행
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    percentages = predictions * 100

    # 선택된 감정 필터링
    selected_emotions = [(i, emotion_labels[i][0]) for i, p in enumerate(percentages.squeeze()) if p >= 30]
    selected_emotions = sorted(selected_emotions, key=lambda x: -percentages.squeeze()[x[0]])[:2]
    if len(selected_emotions) > 1 and percentages.squeeze()[selected_emotions[0][0]] >= 60:
        selected_emotions = [selected_emotions[0]]

    st.write("예측된 감정:")
    for idx, label in selected_emotions:
        st.write(f"{label} ({percentages.squeeze()[idx]:.2f}%) {emotion_labels[idx][1]}")

    # 감정 기반 추천 곡 생성
    emotion_keywords = [emotion_labels[idx][0].lower() for idx, _ in selected_emotions]
    query = " ".join(emotion_keywords)

    # 검색 및 답변 생성
    with st.spinner("추천 곡 검색 중..."):
        retrieved_docs = retriever.invoke(query)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # ChatPromptTemplate 생성
        prompt_template = ChatPromptTemplate.from_messages([
            HumanMessage(content="너는 사용자 감정에 맞는 음악을 추천하는 챗봇이야. 첨부된 문서들을 바탕으로 추천 곡 목록을 제시해줘."
                                 "다만 추천할 수 있는 곡이 없다면 답하지 말아줘.\n"
                                 f"Emotion: {query}\nContext: {format_docs(retrieved_docs)}")
        ])

        # 사용자 지시사항에 맞춰 프롬프트 생성
        user_prompt = prompt_template.format()
        llm = ChatOpenAI(model="gpt-4o")  # LLM 모델 초기화

        # 모델 응답 생성
        response = llm.invoke(user_prompt)

    # 답변 출력 및 세션에 저장
    with st.chat_message("assistant"):
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
