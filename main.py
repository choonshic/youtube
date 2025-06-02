# app.py
import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
from datetime import datetime
from transformers import pipeline

# 댓글 수집 함수
def get_comments(youtube_url, api_key):
    try:
        video_id = youtube_url.split("v=")[-1].split("&")[0]
        youtube = build("youtube", "v3", developerKey=api_key)
        comments, timestamps = [], []
        next_page_token = None

        while True:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()

            for item in response["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(snippet["textDisplay"])
                timestamps.append(snippet["publishedAt"])

            if "nextPageToken" in response:
                next_page_token = response["nextPageToken"]
            else:
                break

        return comments, timestamps
    except HttpError as e:
        st.error("❌ YouTube API 요청 중 오류가 발생했습니다. API 키 또는 URL을 확인해주세요.")
        return [], []

# 형태소 분석
@st.cache_data
def extract_nouns(comments):
    okt = Okt()
    nouns = []
    for comment in comments:
        nouns += okt.nouns(comment)
    return [noun for noun in nouns if len(noun) > 1]

# 감성 분석
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def run_sentiment_analysis(comments, model):
    return model(comments)

# 스트림릿 앱
st.title("YouTube 댓글 분석기")

youtube_url = st.text_input("YouTube 영상 URL 입력")
api_key = st.text_input("API 키 입력")  # type="password" 제거
submit = st.button("분석 시작")

if submit and youtube_url and api_key:
    with st.spinner("댓글 수집 중..."):
        comments, timestamps = get_comments(youtube_url, api_key)
        if not comments:
            st.stop()
        df = pd.DataFrame({"comment": comments, "timestamp": timestamps})
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    with st.spinner("형태소 분석 중..."):
        nouns = extract_nouns(comments)
        word_freq = Counter(nouns)
        df_freq = pd.DataFrame(word_freq.items(), columns=["단어", "빈도수"]).sort_values(by="빈도수", ascending=False)

    st.subheader("워드 클라우드")
    wc = WordCloud(font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", background_color="white", width=800, height=400)
    wc.generate_from_frequencies(word_freq)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("단어 빈도수 상위 20개")
    st.dataframe(df_freq.head(20))
    st.bar_chart(df_freq.head(20).set_index("단어"))

    st.subheader("시간대별 댓글 수")
    df["hour"] = df["timestamp"].dt.hour
    hourly_counts = df.groupby("hour").size().reset_index(name="댓글 수")
    fig2 = px.bar(hourly_counts, x="hour", y="댓글 수", title="시간대별 댓글 빈도")
    st.plotly_chart(fig2)

    st.subheader("댓글 감성 분석")
    model = load_sentiment_model()
    results = run_sentiment_analysis(comments[:100], model)  # API 제한 대비 100개만
    df_sentiment = pd.DataFrame(results)
    df_sentiment["comment"] = comments[:100]
    st.dataframe(df_sentiment[["comment", "label", "score"]])
