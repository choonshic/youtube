# app.py
import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
from datetime import datetime
import re
from soynlp.tokenizer import RegexTokenizer
import os

# 댓글 수집 함수

def extract_video_id(url):
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([\w-]+)",
        r"(?:https?://)?youtu\.be/([\w-]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_comments(youtube_url, api_key):
    try:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("❌ 유효한 YouTube 영상 URL이 아닙니다.")
            return [], []

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
        error_content = e.content.decode("utf-8") if hasattr(e, "content") else str(e)
        st.error("❌ YouTube API 요청 중 오류가 발생했습니다. 오류 메시지를 확인하세요.")
        st.code(error_content, language="json")
        return [], []

# 형태소 분석 (JVM 없이 동작하는 soynlp 기반)
@st.cache_data
def extract_nouns(comments):
    tokenizer = RegexTokenizer()
    nouns = []
    for comment in comments:
        nouns += tokenizer.tokenize(comment)
    return [word for word in nouns if len(word) > 1]

# 스트림릿 앱
st.title("YouTube 댓글 분석기")

youtube_url = st.text_input("YouTube 영상 URL 입력")
api_key = st.text_input("API 키 입력")
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
    # 프로젝트 내 포함된 NanumGothicCoding 폰트 파일 사용
    font_path = os.path.join("fonts", "NanumGothicCoding.ttf")
    if not os.path.exists(font_path):
        st.warning("⚠️ 'fonts/NanumGothicCoding.ttf' 경로에 폰트 파일이 없습니다. 한글이 깨질 수 있습니다.")
        font_path = None
    try:
        wc = WordCloud(font_path=font_path, background_color="white", width=800, height=400)
        wc.generate_from_frequencies(word_freq)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except OSError as e:
        st.error("❌ 워드클라우드 생성 중 오류 발생: 폰트 파일이 유효하지 않거나 로드할 수 없습니다.")
        st.code(str(e))

    st.subheader("단어 빈도수 상위 20개")
    st.dataframe(df_freq.head(20))
    st.bar_chart(df_freq.head(20).set_index("단어"))

    st.subheader("시간대별 댓글 수")
    df["hour"] = df["timestamp"].dt.hour
    hourly_counts = df.groupby("hour").size().reset_index(name="댓글 수")
    fig2 = px.bar(hourly_counts, x="hour", y="댓글 수", title="시간대별 댓글 빈도")
    st.plotly_chart(fig2)
