'''Best Lanaguage Detector yet using TF/IDF vectorizer and SVM'''

_detector_pipeline = None
class LanguageDetector():
    '''
    Best Language Detector in the experiments done in notebooks/language_detection.ipynb
    Uses an N-gram character TF/IDF vectorizer and a Linear SVM classifier.
    '''
    def __init__(self):
        global _detector_pipeline
        if _detector_pipeline:
            self.pipeline = _detector_pipeline
            return

        from pathlib import Path
        model_name="char_n-Gram_svm.joblib"
        model_path = Path(__file__).parent / "models" / model_name
        if model_path.is_file():
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

            self.train()
            _detector_pipeline = self.pipeline
            joblib.dump(self.pipeline, model_path)

    def train(self, texts = None, labels = None):
        if not texts and not labels:
            from datasets import load_dataset
            dataset = load_dataset("papluca/language-identification")
            dataset['train'].shuffle(seed=42)
            texts = dataset['train']['text']
            labels = dataset['train']['labels']
        self.pipeline.fit(texts, labels)

    def predict(self, texts):
        return self.pipeline.predict(list(texts))
