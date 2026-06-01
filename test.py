if __name__ == "__main__":
    from language_detector import CharNGramSVMDetector
    lang_detector = CharNGramSVMDetector()
    predictions = lang_detector.predict(["صبح يبوعمو", "Hi"])
    print(predictions)