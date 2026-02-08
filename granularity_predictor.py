import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "granularity_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

def predict_granularity(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    return {
        "granularity": model.config.id2label[pred_id],
        "confidence": round(float(probs[pred_id]), 3),
    }

if __name__ == "__main__":
    # 3 prompts per granularity label, not seen in granularity_train.jsonl or granularity_valid.jsonl
    tests = [
        ("aggregate", "What is the average deal size by sales region?"),
        ("aggregate", "Show the total number of active sessions per hour."),
        ("aggregate", "Give me the breakdown of expenses by cost center."),
        ("record_level", "Show me the full record for visit ID 8821."),
        ("record_level", "List all consultants in the Boston office with their emails."),
        ("record_level", "Get the detailed profile for appointment with ID A-334."),
    ]
    for expected, prompt in tests:
        result = predict_granularity(prompt)
        print(f"{expected}: {prompt!r} -> {result}")