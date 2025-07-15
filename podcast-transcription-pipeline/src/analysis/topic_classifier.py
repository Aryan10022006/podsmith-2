from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json

class TopicClassifier:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = KMeans(n_clusters=self.num_topics)

    def fit(self, documents):
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.model.fit(tfidf_matrix)

    def predict(self, documents):
        tfidf_matrix = self.vectorizer.transform(documents)
        return self.model.predict(tfidf_matrix)

    def get_topics(self):
        order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names_out()
        topics = {}
        for i in range(self.num_topics):
            topics[i] = [terms[ind] for ind in order_centroids[i, :10]]
        return topics

    def classify_segments(self, segments):
        predictions = self.predict(segments)
        topics = self.get_topics()
        classified_segments = []
        for i, segment in enumerate(segments):
            classified_segments.append({
                "text": segment,
                "predicted_topic": topics[predictions[i]]
            })
        return classified_segments

def save_classified_segments(classified_segments, output_path):
    with open(output_path, 'w') as f:
        json.dump(classified_segments, f)