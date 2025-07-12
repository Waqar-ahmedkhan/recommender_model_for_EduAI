import pandas as pd
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EduAIRecommenderSystem")

app = FastAPI()

# Load pre-trained model and data
try:
    vectorizer = joblib.load("vectorizer.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    data = joblib.load("courses_data.pkl")
    logger.info("Model and data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.post("/api/v1/recommend-courses")
async def recommend_courses(name: str, tags: str):
    try:
        # Split tags into list
        tag_list = [tag.strip() for tag in tags.split(",")]
        query = f"{name} {' '.join(tag_list)}"
        query_vec = vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = cosine_sim.argsort()[::-1][:5]  # Top 5 recommendations
        recommendations = data.iloc[top_indices][['title', 'category', 'level', 'description', 'tags']].to_dict(orient='records')
        return {"success": True, "data": recommendations}
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return {"success": False, "message": "Failed to generate recommendations"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)