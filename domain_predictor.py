import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DOMAIN_MODEL_DIR = "domain_model"

tokenizer = AutoTokenizer.from_pretrained(DOMAIN_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(DOMAIN_MODEL_DIR)
model.eval()

def predict_domain(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    label = model.config.id2label[pred_id]
    conf = float(probs[pred_id])

    return {"domain": label, "confidence": round(conf, 3)}

if __name__ == "__main__":
    print(predict_domain("Show me 1 employee with all info"))