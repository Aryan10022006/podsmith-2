from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import joblib

class EmotionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    @classmethod
    def load_model(cls, filepath):
        model = joblib.load(filepath)
        return cls(model=model)

# Example of a specific emotion detection model
class SimpleEmotionDetector(EmotionModel):
    def __init__(self):
        from sklearn.naive_bayes import MultinomialNB
        super().__init__(model=MultinomialNB())

# Additional models can be defined here as needed.