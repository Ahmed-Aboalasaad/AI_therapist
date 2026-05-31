if __name__ == "__main__":
    from language_detector import CharNGramSVMDetector
    lang_detector = CharNGramSVMDetector()
    predictions = lang_detector.predict(["This is a test sentence.", "C'est une phrase de test."])
    print(predictions)