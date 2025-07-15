from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

class TopicModel:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit(self, documents):
        X = self.vectorizer.fit_transform(documents)
        self.model.fit(X)

    def predict(self, documents):
        X = self.vectorizer.transform(documents)
        return self.model.predict(X)

    def get_topics(self):
        order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names_out()
        topics = {}
        for i in range(self.n_clusters):
            topics[i] = [terms[ind] for ind in order_centroids[i, :10]]
        return topics

    def transform(self, documents):
        X = self.vectorizer.transform(documents)
        return self.model.predict(X)