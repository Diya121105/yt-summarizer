import sqlite3
from datetime import datetime
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

# Database setup
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        url TEXT,
        summary TEXT,
        sentiment TEXT,
        score REAL,
        analyzed_at TEXT
    )''')
    conn.commit()
    conn.close()

def save_to_db(video_id, url, summary, sentiment, score):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (video_id, url, summary, sentiment, score, analyzed_at) VALUES (?, ?, ?, ?, ?, ?)",
              (video_id, url, summary, sentiment, score, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT video_id, url, summary, sentiment, score, analyzed_at FROM history ORDER BY id DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

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
    chunk = transcript[:3000]
    result = groq_call(f"""Analyze the sentiment of this transcript and respond in exactly this format:
SENTIMENT: [Positive/Negative/Neutral]
SCORE: [a number between -1.0 and 1.0]
REASON: [one sentence explanation]

Transcript:
{chunk}""")
    
    lines = result.strip().split("\n")
    sentiment = "Neutral"
    score = 0.0
    reason = ""
    
    for line in lines:
        if line.startswith("SENTIMENT:"):
            sentiment = line.replace("SENTIMENT:", "").strip()
        elif line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except:
                score = 0.0
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    
    return sentiment, score, reason

def get_topics(transcript):
    chunks = chunk_text(transcript, chunk_size=500)
    if len(chunks) < 2:
        chunks = chunks * 2
    vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
    vectorizer.fit_transform(chunks)
    return vectorizer.get_feature_names_out().tolist()

from fpdf import FPDF

def generate_pdf(url, summary, keypoints, sentiment, score, reason, topics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "YouTube Video Analysis Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"URL: {url}", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 7, summary)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Key Points", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 7, keypoints)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Sentiment Analysis", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 7, f"Sentiment: {sentiment} (Score: {score})", ln=True)
    pdf.multi_cell(0, 7, f"Reason: {reason}")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Main Topics", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 7, ", ".join(topics))
    
    return pdf.output()

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

# Compute all results before tabs
    with st.spinner("Analyzing video..."):
        summary = get_summary(transcript)
        keypoints = get_keypoints(transcript)
        label, score, reason = get_sentiment(transcript)
        topics = get_topics(transcript)
    
    save_to_db(video_id, url, summary, label, score)

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔑 Key Points", "😊 Sentiment", "🏷️ Topics"])

    with tab1:
        st.subheader("Video Summary")
        st.write(summary)

    with tab2:
        st.subheader("Key Points")
        st.write(keypoints)

    with tab3:
        st.subheader("Sentiment Analysis")
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", label)
        col2.metric("Polarity Score", score)
        st.info(f"💡 {reason}")
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
        st.subheader("Main Topics")
        for i, topic in enumerate(topics, 1):
            st.markdown(f"**{i}.** {topic}")
            
# PDF Export
    st.divider()
    st.subheader("📄 Export Full Report")
    pdf_bytes = generate_pdf(url, summary, keypoints, label, score, reason, topics)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=bytes(pdf_bytes),
        file_name="video_analysis.pdf",
        mime="application/pdf"
    )

if st.button("Compare Videos"):
    if not url1 or not url2:
        st.error("Please enter both URLs")
        st.stop()
    
    vid1 = extract_video_id(url1)
    vid2 = extract_video_id(url2)
    
    with st.spinner("Analyzing both videos..."):
        t1 = get_transcript(vid1)
        t2 = get_transcript(vid2)
        
        if not t1 or not t2:
            st.error("One or both videos have no transcript")
            st.stop()
        
        s1 = get_summary(t1)
        s2 = get_summary(t2)
        sent1, score1, reason1 = get_sentiment(t1)
        sent2, score2, reason2 = get_sentiment(t2)
        topics1 = get_topics(t1)
        topics2 = get_topics(t2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎬 Video 1")
        st.write(s1)
        st.metric("Sentiment", sent1, score1)
        st.write("**Topics:** " + ", ".join(topics1[:5]))
    with col2:
        st.markdown("### 🎬 Video 2")
        st.write(s2)
        st.metric("Sentiment", sent2, score2)
        st.write("**Topics:** " + ", ".join(topics2[:5]))
    
    # Sentiment comparison chart
    fig = px.bar(
        x=["Video 1", "Video 2"],
        y=[score1, score2],
        color=[score1, score2],
        color_continuous_scale="RdYlGn",
        range_y=[-1, 1],
        title="Sentiment Comparison",
        labels={"y": "Polarity Score", "x": ""}
    )
    st.plotly_chart(fig, use_container_width=True)

    # History Section
st.divider()
st.subheader("📚 Recent Analyses")
history = get_history()
if history:
    for row in history:
        with st.expander(f"🎬 {row[1]} — {row[5]}"):
            st.write(f"**Summary:** {row[2]}")
            st.write(f"**Sentiment:** {row[3]} ({row[4]})")
else:
    st.info("No analyses yet — analyze a video to see history here!")          