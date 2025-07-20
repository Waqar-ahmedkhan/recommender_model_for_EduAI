import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Course data (including MDCAT courses, with price, duration, thumbnail, and enhanced tags)
courses = [
    # --- AI & Machine Learning ---
    {
        "id": "687cdb5f777ad3709aa023f7",
        "title": "AI Basics for Beginners",
        "thumbnail": "/images/ai-basics.jpg",
        "category": "AI & ML",
        "level": "Beginner",
        "description": "Understand the fundamentals of artificial intelligence and real-world use cases.",
        "tags": "ai, machine learning, artificial intelligence, basics",
        "rating": 4.3,
        "price": 99,
        "duration": 300
    },
    {
        "id": "687cdcef777ad3709aa02416",
        "title": "Intermediate Machine Learning",
        "thumbnail": "/images/ml-intermediate.jpeg",
        "category": "AI & ML",
        "level": "Intermediate",
        "description": "Dive into supervised learning, decision trees, and model evaluation techniques.",
        "tags": "machine learning, decision trees, sklearn, ai",
        "rating": 4.5,
        "price": 129,
        "duration": 360
    },
    {
        "id": "687cded8777ad3709aa0245c",
        "title": "Advanced Deep Learning with PyTorch",
        "thumbnail": "/images/deep-learning.png",
        "category": "AI & ML",
        "level": "Advanced",
        "description": "Build and train deep neural networks using PyTorch for real-world applications.",
        "tags": "deep learning, pytorch, ai, neural networks",
        "rating": 4.7,
        "price": 149,
        "duration": 480
    },
    # --- Web Development ---
    {
        "id": "687ce21a777ad3709aa02469",
        "title": "HTML, CSS & JS Starter Pack",
        "thumbnail": "/images/web-basic.png",
        "category": "Web Development",
        "level": "Beginner",
        "description": "Learn to build beautiful websites using HTML, CSS, and JavaScript.",
        "tags": "html, css, javascript, frontend, web, development, beginner",
        "rating": 4.4,
        "price": 79,
        "duration": 240
    },
    {
        "id": "5",
        "title": "React.js & TypeScript Masterclass",
        "thumbnail": "/images/react-ts.jpeg",
        "category": "Web Development",
        "level": "Intermediate",
        "description": "Build dynamic frontend apps using React and TypeScript with hooks.",
        "tags": "react, typescript, frontend, components, web, development",
        "rating": 4.6,
        "price": 119,
        "duration": 400
    },
    {
        "id": "6",
        "title": "Full Stack with Next.js & MongoDB",
        "thumbnail": "/images/fullstack.png",
        "category": "Web Development",
        "level": "Advanced",
        "description": "Create full-stack applications with authentication and APIs using Next.js.",
        "tags": "next.js, mongodb, fullstack, api, devops, web, development",
        "rating": 4.8,
        "price": 159,
        "duration": 500
    },
    # --- NTS / CSS / NET ---
    {
        "id": "7",
        "title": "CSS Essay Writing Crash Course",
        "thumbnail": "/images/css.jpeg",
        "category": "CSS",
        "level": "Advanced",
        "description": "Learn structured, impactful essay writing techniques for CSS preparation.",
        "tags": "css, essay writing, english",
        "rating": 4.8,
        "price": 89,
        "duration": 200
    },
    {
        "id": "687ce349777ad3709aa0247b",
        "title": "NUST NET Physics Prep",
        "thumbnail": "/images/net-physics.jpeg",
        "category": "NET",
        "level": "Intermediate",
        "description": "Master NET Physics concepts with practice questions and conceptual videos.",
        "tags": "net, physics, nust",
        "rating": 4.6,
        "price": 99,
        "duration": 320
    },
    {
        "id": "687ce6a7777ad3709aa024e1",
        "title": "NTS English Grammar Bootcamp",
        "thumbnail": "/images/nts-english.jpeg",
        "category": "NTS",
        "level": "Beginner",
        "description": "Improve your grammar, tenses, and comprehension for the NTS exam.",
        "tags": "nts, english, grammar",
        "rating": 4.3,
        "price": 69,
        "duration": 180
    },
    # --- Coding Core Skills ---
    {
        "id": "10",
        "title": "Learn Python Programming from Scratch",
        "thumbnail": "/images/python-basics.jpeg",
        "category": "Programming",
        "level": "Beginner",
        "description": "Start programming with Python by mastering syntax, loops, functions, and more.",
        "tags": "python, programming, loops, functions",
        "rating": 4.5,
        "price": 89,
        "duration": 280
    },
    {
        "id": "11",
        "title": "OOP in Java with Real Projects",
        "thumbnail": "/images/java-oop.jpeg",
        "category": "Programming",
        "level": "Intermediate",
        "description": "Understand object-oriented programming in Java with hands-on projects.",
        "tags": "java, oop, classes, objects, inheritance",
        "rating": 4.4,
        "price": 109,
        "duration": 350
    },
    {
        "id": "12",
        "title": "Data Structures & Algorithms Bootcamp",
        "thumbnail": "/images/dsa.jpg",
        "category": "Programming",
        "level": "Advanced",
        "description": "Master sorting, searching, trees, graphs and algorithms for coding interviews.",
        "tags": "algorithms, data structures, competitive programming",
        "rating": 4.9,
        "price": 169,
        "duration": 600
    },
    # --- Extra AI Courses ---
    {
        "id": "13",
        "title": "Build an AI Chatbot with Python",
        "thumbnail": "/images/ai-chatbot.jpeg",
        "category": "AI & ML",
        "level": "Advanced",
        "description": "Use NLP and machine learning to create a smart chatbot with Python.",
        "tags": "nlp, chatbot, ai, python",
        "rating": 4.7,
        "price": 139,
        "duration": 420
    },
    {
        "id": "14",
        "title": "Math for Machine Learning",
        "thumbnail": "/images/ml-math.jpeg",
        "category": "AI & ML",
        "level": "Intermediate",
        "description": "Learn linear algebra, probability, and calculus used in machine learning.",
        "tags": "math, machine learning, linear algebra, calculus",
        "rating": 4.5,
        "price": 119,
        "duration": 360
    },
    # --- Soft Skills / General ---
    {
        "id": "15",
        "title": "Technical Interview Preparation",
        "thumbnail": "/images/interview-prep.jpg",
        "category": "Career Skills",
        "level": "Intermediate",
        "description": "Practice behavioral and coding interview questions with mock sessions.",
        "tags": "interview, career, soft skills, preparation",
        "rating": 4.6,
        "price": 99,
        "duration": 200
    },
    {
        "id": "16",
        "title": "Freelancing 101 for Developers",
        "thumbnail": "/images/freelancing.jpg",
        "category": "Career Skills",
        "level": "Beginner",
        "description": "Learn how to build a freelance profile, win clients, and manage projects.",
        "tags": "freelancing, career, fiverr, upwork",
        "rating": 4.4,
        "price": 79,
        "duration": 180
    },
    {
        "id": "17",
        "title": "Git & GitHub Essentials",
        "thumbnail": "/images/git-github.jpg",
        "category": "Tools",
        "level": "Beginner",
        "description": "Learn version control and collaboration using Git and GitHub.",
        "tags": "git, github, version control, open source",
        "rating": 4.2,
        "price": 69,
        "duration": 150
    },
    {
        "id": "18",
        "title": "REST APIs with Node.js",
        "thumbnail": "/images/node-api.jpg",
        "category": "Web Development",
        "level": "Intermediate",
        "description": "Create fast and secure RESTful APIs using Node, Express, and MongoDB.",
        "tags": "node.js, api, rest, backend, web, development",
        "rating": 4.7,
        "price": 129,
        "duration": 380
    },
    {
        "id": "19",
        "title": "DevOps with Docker & CI/CD",
        "thumbnail": "/images/devops.jpg",
        "category": "DevOps",
        "level": "Advanced",
        "description": "Learn containerization, pipelines, and CI/CD workflows for deployment.",
        "tags": "docker, devops, ci/cd, deployment",
        "rating": 4.6,
        "price": 149,
        "duration": 450
    },
    {
        "id": "20",
        "title": "Prompt Engineering for AI",
        "thumbnail": "/images/prompt-ai.jpeg",
        "category": "AI & ML",
        "level": "Advanced",
        "description": "Learn how to craft powerful prompts for GPT and other language models.",
        "tags": "gpt, prompt engineering, llm, ai",
        "rating": 4.8,
        "price": 139,
        "duration": 400
    },
    # --- MDCAT Courses ---
    {
        "id": "21",
        "title": "MDCAT Beginner Prep",
        "thumbnail": "/images/mdcat-basics.jpeg",
        "category": "MDCAT",
        "level": "Beginner",
        "description": "Foundational MDCAT preparation with video lectures and quizzes, aligned with PMDC 2025 syllabus.",
        "tags": "mdcat, medical, entrance exam, biology, chemistry, physics",
        "rating": 4.4,
        "price": 89,
        "duration": 300
    },
    {
        "id": "22",
        "title": "MDCAT Intermediate Biology & Chemistry",
        "thumbnail": "/images/mdcat-intermediate.jpeg",
        "category": "MDCAT",
        "level": "Intermediate",
        "description": "In-depth study of biology and chemistry for MDCAT with practice tests and conceptual clarity.",
        "tags": "mdcat, biology, chemistry, medical",
        "rating": 4.6,
        "price": 109,
        "duration": 400
    },
    {
        "id": "23",
        "title": "MDCAT Advanced Practice & Mock Tests",
        "thumbnail": "/images/mdcat-advanced.jpg",
        "category": "MDCAT",
        "level": "Advanced",
        "description": "Rigorous MDCAT preparation with full-length mock tests and advanced problem-solving.",
        "tags": "mdcat, mock tests, medical, advanced",
        "rating": 4.7,
        "price": 129,
        "duration": 450
    }
]

# Process data
df = pd.DataFrame(courses)
df["combined"] = df["title"] + " " + df["tags"] + " " + df["description"]

# Create TF-IDF vectorizer with adjusted parameters
vectorizer = TfidfVectorizer(stop_words=None, lowercase=True, token_pattern=r'(?u)\b\w+\b')
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# Save the model and data
joblib.dump(df, "courses_data.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


# Recommendation function for testing
def recommend_courses(query, top_n=5, category=None, level=None):
    """
    Recommend courses based on a text query using cosine similarity.

    Args:
        query (str): User's search query
        top_n (int): Number of recommendations to return
        category (str, optional): Filter by category
        level (str, optional): Filter by level

    Returns:
        pd.DataFrame: Recommended courses
    """
    query_vector = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1]
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity'] = similarities[top_indices]

    if category:
        recommendations = recommendations[recommendations['category'] == category]
    if level:
        recommendations = recommendations[recommendations['level'] == level]

    return recommendations[
        ['id', 'title', 'category', 'level', 'description', 'tags', 'rating', 'price', 'duration', 'thumbnail',
         'similarity']].head(top_n)


# Example usage
if __name__ == "__main__":
    print("✅ Model trained and saved successfully!")
    query1 = "machine learning python"
    query2 = "web development beginner"
    query3 = "mdcat biology"
    print("\nExample Recommendations for 'machine learning python':")
    print(recommend_courses(query1, top_n=3))
    print("\nExample Recommendations for 'web development beginner' with category 'Web Development':")
    print(recommend_courses(query2, top_n=3, category="Web Development"))
    print("\nExample Recommendations for 'mdcat biology' with category 'MDCAT':")
    print(recommend_courses(query3, top_n=3, category="MDCAT"))