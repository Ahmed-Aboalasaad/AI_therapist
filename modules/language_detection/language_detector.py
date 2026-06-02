'''Best Lanaguage Detector using TF/IDF vectorizer and SVM'''

class LanguageDetector():
    '''
    Best Language Detector in the experiments done in notebooks/language_detection.ipynb
    Uses an N-gram character TF/IDF vectorizer and a Linear SVM classifier.'''

    def __init__(self, model_name="char_n-Gram_svm.joblib"):
        if model_name:
            import joblib
            from pathlib import Path
            BASE_DIR = Path(__file__).parent
            self.pipeline = joblib.load(BASE_DIR / "models" / model_name)
        else:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC

            self.pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(analyzer="char", ngram_range=(2, 4))),
                ("classifier", LinearSVC())
            ])

    def train(self, texts, labels):
        self.pipeline.fit(texts, labels)

    def predict(self, texts):
        return self.pipeline.predict(list(texts))
