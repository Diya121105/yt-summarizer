import requests
import streamlit as st
import plotly.express as px

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="YouTube Summarizer", layout="wide")
st.title("🎬 YouTube Video Analyzer")
st.markdown("Paste any YouTube URL to get AI-powered insights")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Video"):
    if not url:
        st.error("Please enter a YouTube URL")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔑 Key Points", "😊 Sentiment", "🏷️ Topics"])

    with tab1:
        with st.spinner("Generating summary..."):
            try:
                res = requests.post(f"{API_BASE}/summarize", json={"url": url})
                data = res.json()
                st.subheader("Video Summary")
                st.write(data["summary"])
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        with st.spinner("Extracting key points..."):
            try:
                res = requests.post(f"{API_BASE}/keypoints", json={"url": url})
                data = res.json()
                st.subheader("Key Points")
                st.write(data["keypoints"])
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        with st.spinner("Analyzing sentiment..."):
            try:
                res = requests.post(f"{API_BASE}/sentiment", json={"url": url})
                data = res.json()
                st.subheader("Sentiment Analysis")

                col1, col2, col3 = st.columns(3)
                col1.metric("Sentiment", data["sentiment"])
                col2.metric("Polarity Score", data["polarity_score"])
                col3.metric("Sentences Analyzed", data["sentence_count"])

                fig = px.bar(
                    x=["Polarity Score"],
                    y=[data["polarity_score"]],
                    color=[data["polarity_score"]],
                    color_continuous_scale="RdYlGn",
                    range_y=[-1, 1],
                    labels={"y": "Score", "x": ""}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab4:
        with st.spinner("Extracting topics..."):
            try:
                res = requests.post(f"{API_BASE}/topics", json={"url": url})
                data = res.json()
                st.subheader("Main Topics")
                topics = data["topics"]
                for i, topic in enumerate(topics, 1):
                    st.markdown(f"**{i}.** {topic}")
            except Exception as e:
                st.error(f"Error: {e}")