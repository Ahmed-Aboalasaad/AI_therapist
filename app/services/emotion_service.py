import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.core.constants import EMOTION_MAP

MODEL_PATH = "ahmedsharaf/emotion-classifier"
MAX_LENGTH = 128

class EmotionService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence = probs.max().item()
            pred_idx = probs.argmax().item()

        return {
            "emotion": EMOTION_MAP[pred_idx],
            "confidence": round(confidence, 4)
        }
