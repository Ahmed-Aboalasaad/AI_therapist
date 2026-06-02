'''Best Lanaguage Detector using TF/IDF vectorizer and SVM'''

from datasets import load_dataset
_detector_pipeline = None

class LanguageDetector():
    '''
    Best Language Detector in the experiments done in notebooks/language_detection.ipynb
    Uses an N-gram character TF/IDF vectorizer and a Linear SVM classifier.'''

    def __init__(self, model_name=None):
        global _detector_pipeline
        if not model_name and _detector_pipeline is not None:
            self.pipeline = _detector_pipeline
            return

        if model_name:
            import joblib
            from pathlib import Path
            BASE_DIR = Path(__file__).parent
            self.pipeline = joblib.load(BASE_DIR / "models" / model_name)
        else:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC
            from datasets import load_dataset

            self.pipeline = Pipeline([
                ("vectorizer", TfidfVectorizer(analyzer="char", ngram_range=(2, 4))),
                ("classifier", LinearSVC())
            ])

            self.train()
            _detector_pipeline = self.pipeline

    def train(self, texts = None, labels = None):
        if  not texts and not labels:
            print("load data")
            dataset = load_dataset("papluca/language-identification")
            print("loaded data")
            dataset = dataset['train']
            dataset.shuffle(seed=42)
            texts = dataset['text']
            labels = dataset['labels']

        print("start language model traning")
        self.pipeline.fit(texts, labels)
        print("finish language model traning")

    def predict(self, texts):
        return self.pipeline.predict(list(texts))
