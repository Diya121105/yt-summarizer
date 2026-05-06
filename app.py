import requests
import streamlit as st
import plotly.express as px
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from groq import Groq
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

# Cache
cache = {}

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    if video_id in cache:
        return cache[video_id]
    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id)
        text = " ".join([t.text for t in transcript])
        cache[video_id] = text
        return text
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def chunk_text(text, chunk_size=3000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        current_chunk.append(word)
        current_size += len(word)
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def groq_call(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

def get_summary(transcript):
    chunks = chunk_text(transcript)
    chunk_summaries = []
    for chunk in chunks[:5]:
        summary = groq_call(f"Summarize this transcript excerpt concisely:\n{chunk}")
        chunk_summaries.append(summary)
    return groq_call(f"Combine these into one clear summary:\n" + "\n".join(chunk_summaries))

def get_keypoints(transcript):
    chunks = chunk_text(transcript)
    return groq_call(f"Extract 7 key points as bullet points:\n{chunks[0]}")

def get_sentiment(transcript):
    sentences = transcript.split(".")[:100]
    polarities = [TextBlob(s).sentiment.polarity for s in sentences if s.strip()]
    avg = sum(polarities) / len(polarities) if polarities else 0
    label = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
    return label, round(avg, 3), len(polarities)

def get_topics(transcript):
    chunks = chunk_text(transcript, chunk_size=500)
    if len(chunks) < 2:
        chunks = chunks * 2
    vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
    vectorizer.fit_transform(chunks)
    return vectorizer.get_feature_names_out().tolist()

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube Analyzer", layout="wide")
st.title("🎬 YouTube Video Analyzer")
st.markdown("Paste any YouTube URL to get AI-powered insights")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Video"):
    if not url:
        st.error("Please enter a YouTube URL")
        st.stop()

    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    with st.spinner("Fetching transcript..."):
        transcript = get_transcript(video_id)
        if not transcript:
            st.error("No transcript available for this video")
            st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔑 Key Points", "😊 Sentiment", "🏷️ Topics"])

    with tab1:
        with st.spinner("Generating summary..."):
            summary = get_summary(transcript)
            st.subheader("Video Summary")
            st.write(summary)

    with tab2:
        with st.spinner("Extracting key points..."):
            keypoints = get_keypoints(transcript)
            st.subheader("Key Points")
            st.write(keypoints)

    with tab3:
        with st.spinner("Analyzing sentiment..."):
            label, score, count = get_sentiment(transcript)
            st.subheader("Sentiment Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", label)
            col2.metric("Polarity Score", score)
            col3.metric("Sentences Analyzed", count)
            fig = px.bar(
                x=["Polarity Score"],
                y=[score],
                color=[score],
                color_continuous_scale="RdYlGn",
                range_y=[-1, 1],
                labels={"y": "Score", "x": ""}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        with st.spinner("Extracting topics..."):
            topics = get_topics(transcript)
            st.subheader("Main Topics")
            for i, topic in enumerate(topics, 1):
                st.markdown(f"**{i}.** {topic}")