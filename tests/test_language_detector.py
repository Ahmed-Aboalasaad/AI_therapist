if __name__ == "__main__":
    from modules.language_detection.language_detector import LanguageDetector

    print("Easy:")
    print(LanguageDetector().predict([
        "Hello this is a test",
        "استر يا دولى",
        "Bonjour tous le monde!",
        "Hola a todos!",
        "Привет!"
    ]))
    print("['en' 'ar' 'fr' 'es' 'ru']")

    print("\nHard:")
    print(LanguageDetector().predict([
        "Hello",
        "Hola",
    ]))
    print("['en' 'es']")