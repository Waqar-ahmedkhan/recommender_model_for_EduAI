# 📊 EduAI – ML-Based Course Recommendation System

This repository contains the **Python machine learning model** developed for the [EduAI](https://github.com/Waqar-ahmedkhan/eduai) platform. The model is designed to provide **intelligent course recommendations** to students based on course similarity using **cosine similarity** and **content-based filtering**.

---

## 🚀 Features

* 🔍 **Content-based filtering** for personalized course recommendations
* 🧠 **Cosine similarity** model trained on course metadata
* 📈 Easy integration with EduAI’s backend API (Node.js)
* 📊 Recommendation engine that returns top N similar courses
* 🧪 Ready-to-train and test with your custom course dataset

---

## 🧠 Machine Learning Model Details

### 🔹 Approach

* Type: **Content-Based Recommendation**
* Algorithm: **TF-IDF Vectorizer + Cosine Similarity**
* Input: Course dataset (CSV/JSON) with fields like `title`, `description`, `tags`
* Output: Top-N recommended courses for a given input course

### 🔹 Tech Stack

* Python 3.10+
* Scikit-learn
* Pandas
* NumPy

---

## 📦 Installation

```bash
# Clone this repo
git clone https://github.com/Waqar-ahmedkhan/eduai-recommendation.git
cd eduai-recommendation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🗃️ Dataset Format

Your input CSV file should look like this:

```csv
title,description,tags
"Intro to AI","Learn basics of artificial intelligence.","AI, beginner, theory"
"Advanced Python","Master Python with OOP and real projects.","Python, advanced, backend"
...
```

---

## 🏗️ How to Train

```bash
python train_model.py --dataset courses.csv --output model.pkl
```

---

## 🤖 How to Get Recommendations

```bash
python recommend.py --course "Intro to AI" --top_n 5
```

Expected Output:

```json
[
  {"title": "AI for Beginners", "score": 0.94},
  {"title": "Machine Learning Basics", "score": 0.89},
  ...
]
```

---

## 🔁 Integration with EduAI Backend

* Expose your `recommend()` method as a REST API endpoint using **Flask** or **FastAPI**
* Use course name or ID as input, return top-N JSON response
* Call from frontend using Axios

---

## 📌 To-Do / Future Work

* [ ] Improve NLP preprocessing with spaCy or NLTK
* [ ] Integrate user-based collaborative filtering
* [ ] Save training logs and metrics
* [ ] Deploy as microservice using Docker + FastAPI

---

## 🧠 Developed by

**Waqar Ahmed Khan**
Full Stack DevOps Engineer & ML Engineer
📧 [waqarahmed44870@gmail.com](mailto:waqarahmed44870@gmail.com)
🔗 GitHub: [Waqar-ahmedkhan](https://github.com/Waqar-ahmedkhan)

---

## 📄 License

MIT License © 2025 Waqar Ahmed Khan
