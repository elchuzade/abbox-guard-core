import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "intent_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict_intent(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    intent = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    return {
        "intent": intent,
        "confidence": round(confidence, 3)
    }


if __name__ == "__main__":
    prompt = "Show me 3 doctors with all info except EFN"
    print(predict_intent(prompt))