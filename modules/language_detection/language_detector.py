'''Best Lanaguage Detector using TF/IDF vectorizer and SVM'''

class CharNGramSVMDetector():
    def __init__(self, model_path="models/language_detector_svm.joblib"):
        if model_path:
            import joblib
            self.pipeline = joblib.load(model_path)
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
