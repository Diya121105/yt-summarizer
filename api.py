from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from groq import Groq
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

app = FastAPI(title="YouTube Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# In-memory cache
cache = {}

# --- Models ---
class VideoRequest(BaseModel):
    url: str

# --- Helpers ---
def extract_video_id(url: str):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise HTTPException(status_code=400, detail="Invalid YouTube URL")

def get_transcript(video_id: str):
    if video_id in cache:
        return cache[video_id]
    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id)
        text = " ".join([t.text for t in transcript])
        cache[video_id] = text
        return text
    except (TranscriptsDisabled, NoTranscriptFound):
        raise HTTPException(status_code=404, detail="No transcript available for this video")

def chunk_text(text: str, chunk_size: int = 3000):
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

def groq_call(prompt: str):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content

# --- Endpoints ---
@app.post("/summarize")
def summarize(req: VideoRequest):
    video_id = extract_video_id(req.url)
    transcript = get_transcript(video_id)
    chunks = chunk_text(transcript)
    
    chunk_summaries = []
    for chunk in chunks[:5]:  # limit to first 5 chunks
        summary = groq_call(f"Summarize this transcript excerpt concisely:\n{chunk}")
        chunk_summaries.append(summary)
    
    final_summary = groq_call(
        f"Combine these summaries into one clear, well-structured summary:\n" + 
        "\n".join(chunk_summaries)
    )
    return {"video_id": video_id, "summary": final_summary}

@app.post("/keypoints")
def keypoints(req: VideoRequest):
    video_id = extract_video_id(req.url)
    transcript = get_transcript(video_id)
    chunks = chunk_text(transcript)
    
    result = groq_call(
        f"Extract the 7 most important key points from this transcript as bullet points:\n{chunks[0]}"
    )
    return {"video_id": video_id, "keypoints": result}

@app.post("/sentiment")
def sentiment(req: VideoRequest):
    video_id = extract_video_id(req.url)
    transcript = get_transcript(video_id)
    
    sentences = transcript.split(".")[:100]
    polarities = [TextBlob(s).sentiment.polarity for s in sentences if s.strip()]
    
    avg_polarity = sum(polarities) / len(polarities) if polarities else 0
    
    if avg_polarity > 0.1:
        label = "Positive"
    elif avg_polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    
    return {
        "video_id": video_id,
        "sentiment": label,
        "polarity_score": round(avg_polarity, 3),
        "sentence_count": len(polarities)
    }

@app.post("/topics")
def topics(req: VideoRequest):
    video_id = extract_video_id(req.url)
    transcript = get_transcript(video_id)
    chunks = chunk_text(transcript, chunk_size=500)
    
    if len(chunks) < 2:
        chunks = chunks * 2
    
    vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
    vectorizer.fit_transform(chunks)
    top_topics = vectorizer.get_feature_names_out().tolist()
    
    return {"video_id": video_id, "topics": top_topics}

@app.get("/health")
def health():
    return {"status": "ok"}
