import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EduAIRecommenderSystem")

class CourseRecommender:
    def __init__(self, csv_file):
        self.data = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.csv_file = csv_file

    def load_data(self):
        try:
            # Check if the file exists
            if not os.path.isfile(self.csv_file):
                raise FileNotFoundError(f"The file {self.csv_file} does not exist.")
            self.data = pd.read_csv(self.csv_file)
            self.data.dropna(subset=["title", "description"], inplace=True)
            # Generate tags based on category and level since tags column is missing
            self.data["tags"] = self.data.apply(lambda row: f"{row['category']} {row['level']}".lower(), axis=1)
            self.data["combined_features"] = self.data["title"] + " " + self.data["description"] + " " + self.data["tags"]
            logger.info(f"Loaded {len(self.data)} courses.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def build_model(self):
        try:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.data["combined_features"])
            logger.info("TF-IDF model built successfully.")
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise

    def recommend(self, query, top_n=5):
        try:
            query_vec = self.vectorizer.transform([query])
            cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = cosine_sim.argsort()[::-1][:top_n]
            recommendations = self.data.iloc[top_indices][['title', 'category', 'level', 'description', 'tags']]
            return recommendations.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return []

    def save_model(self, vectorizer_path="vectorizer.pkl", matrix_path="tfidf_matrix.pkl", data_path="courses_data.pkl"):
        try:
            joblib.dump(self.vectorizer, vectorizer_path)
            joblib.dump(self.tfidf_matrix, matrix_path)
            joblib.dump(self.data, data_path)
            logger.info("Model and data saved successfully.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

def main():
    print("\n=== Course Recommendation System ===")
    available_datasets = [f for f in os.listdir() if f.endswith('.csv')]
    print(f"Available datasets: {available_datasets}")
    dataset_input = input("Enter dataset path (default: pakistani_courses_dataset_main.csv): ").strip()

    # Validate input and set default if invalid
    dataset = dataset_input if dataset_input in available_datasets else "pakistani_courses_dataset_main.csv"
    if not os.path.isfile(dataset):
        print(f"Warning: {dataset} not found. Using default: pakistani_courses_dataset_main.csv")
        dataset = "pakistani_courses_dataset_main.csv"

    recommender = CourseRecommender(dataset)

    try:
        recommender.load_data()
        recommender.build_model()
        recommender.save_model()
        print("Model trained and saved successfully!")
    except Exception as e:
        print(f"Error loading dataset or building model: {e}")
        return

    while True:
        print("\n=== Menu ===")
        print("1. Get course recommendations")
        print("2. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            query = input("Search by course name or topic (e.g. 'css advanced', 'mdcat', 'issb'): ")
            num = int(input("How many results? (default 5): ") or 5)
            results = recommender.recommend(query, top_n=num)

            if results:
                print("\nTop Recommendations:\n")
                for i, course in enumerate(results, 1):
                    print(f"{i}. {course['title']} ({course['category']} - {course['level']})\n   {course['description']}\n")
            else:
                print("No relevant courses found.")
        elif choice == "2":
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()