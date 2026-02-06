import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "granularity_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict_granularity(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    return {
        "granularity": model.config.id2label[pred_id],
        "confidence": round(float(probs[pred_id]), 3),
    }

if __name__ == "__main__":
    print(predict_granularity("Show average salary by team"))
    print(predict_granularity("List employees with emails"))