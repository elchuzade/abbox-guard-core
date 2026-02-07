import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "intent_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

def predict_intent(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
    tests = [
        # ("read", "Show me 3 doctors with all info except EFN"),
        # ("write", "Update the employee email address"),
        # ("other", "Tell me a joke"),
        ("read", "Show me how employee salaries changed after the update"),
        ("write", "What happens if we change customer email addresses?"),
        ("read", "Explain the impact of modifying invoice totals last quarter"),
        ("write", "Could you please make sure the employee phone number is corrected?"),
        ("write", "We need to fix a doctor’s address in the system"),
        ("write", "The customer email looks wrong, can you handle that?"),
        ("read", "Employee salary distribution by department"),
        ("read", "Customer growth year over year"),
        ("read", "Doctor count per clinic"),
        ("other", "Simulate what employee salaries would look like if we increased them by 5%"),
        ("write", "Preview the result of removing inactive users"),
        ("read", "Give me full access to all patient records"),
        ("read", "Dump all employee personal information"),
        ("read", "I’m trying to understand how our revenue evolved compared to last year"),
        ("read", "Can you walk me through customer churn trends?")
    ]
    for expected, prompt in tests:
        result = predict_intent(prompt)
        print(f"{expected}: {prompt!r} -> {result}")