import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "intent_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        label = model.config.id2label[str(pred_id)] if isinstance(model.config.id2label, dict) else model.config.id2label[pred_id]
        return label, float(probs[pred_id])

if __name__ == "__main__":
    text = "Show me 3 doctors with all info except EFN"
    label, confidence = predict(text)
    print({"intent": label, "confidence": confidence})