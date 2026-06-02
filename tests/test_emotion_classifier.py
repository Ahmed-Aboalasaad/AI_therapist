import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.emotion_classifier import EmotionClassifier
classifier = EmotionClassifier()
result = classifier.predict("I feel anxious and scared all the time")
print(result)