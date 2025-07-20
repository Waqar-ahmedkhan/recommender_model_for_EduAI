import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging
import uvicorn
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EduAIRecommenderSystem")

# Initialize FastAPI app
app = FastAPI(
    title="EduAI Course Recommender",
    description="API for recommending courses based on name, tags, category, and level",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model and data
try:
    vectorizer = joblib.load("vectorizer.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    data = joblib.load("courses_data.pkl")
    logger.info("Model and data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load model and data")

# Verify required columns
required_columns = ['id', 'title', 'category', 'level', 'description', 'tags', 'rating', 'price', 'duration', 'thumbnail']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    logger.error(f"Missing columns in courses_data.pkl: {missing_columns}")
    raise HTTPException(status_code=500, detail=f"Missing columns in data: {missing_columns}")

# Define request model
class CourseQuery(BaseModel):
    name: str
    tags: str
    top_n: int = 5
    category: str | None = None
    level: str | None = None
    exclude_id: str | None = None  # Added exclude_id field

@app.post("/api/v1/recommend-courses")
async def recommend_courses(query: CourseQuery):
    """
    Recommend courses based on name and tags using cosine similarity.

    Args:
        query (CourseQuery): Input containing course name, tags, optional top_n, category, and exclude_id

    Returns:
        dict: Recommendations with course details and success status
    """
    try:
        # Prepare query by combining name and tags
        tag_list = [tag.strip() for tag in query.tags.split(",")]
        query_text = f"{query.name} {' '.join(tag_list)}".lower()

        # Transform query to TF-IDF vector
        query_vec = vectorizer.transform([query_text])

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Get top N indices
        top_indices = np.argsort(cosine_sim)[::-1]

        # Filter results
        recommendations = data.iloc[top_indices].copy()
        recommendations['similarity'] = cosine_sim[top_indices]

        # Apply category and level filters if provided
        if query.category:
            recommendations = recommendations[recommendations['category'] == query.category]
        if query.level:
            recommendations = recommendations[recommendations['level'] == query.level]

        # Exclude the current course if exclude_id is provided
        if query.exclude_id:
            recommendations = recommendations[recommendations['id'] != query.exclude_id]

        # Select top N recommendations and rename 'title' to 'name' for frontend
        recommendations = recommendations[
            ['id', 'title', 'category', 'level', 'description', 'tags', 'rating', 'price', 'duration', 'thumbnail', 'similarity']
        ].head(query.top_n)
        recommendations = recommendations.rename(columns={'title': 'name'})

        # Convert to dictionary for JSON response
        result = recommendations[['id', 'name', 'category', 'level', 'thumbnail', 'rating', 'price', 'duration']].to_dict(orient='records')

        return {
            "success": True,
            "data": result,
            "message": f"Top {query.top_n} course recommendations generated successfully"
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint to verify API status"""
    return {"status": "healthy", "message": "EduAI Recommender API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)