import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DOMAIN_MODEL_DIR = "domain_model"

tokenizer = AutoTokenizer.from_pretrained(DOMAIN_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(DOMAIN_MODEL_DIR)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

def predict_domain(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    label = model.config.id2label[pred_id]
    conf = float(probs[pred_id])

    return {"domain": label, "confidence": round(conf, 3)}

if __name__ == "__main__":
    # 3 prompts per domain, not seen in domain_train.jsonl or domain_valid.jsonl
    tests = [
        ("hr", "Show the probation period end dates for recent hires."),
        ("hr", "List employees who have not completed the mandatory ethics training."),
        ("hr", "Get the job offer letter template for the senior analyst role."),
        ("healthcare", "Show me the CT scan results for the trauma patient."),
        ("healthcare", "List patients scheduled for the dialysis unit tomorrow."),
        ("healthcare", "Retrieve the advance directive on file for the patient."),
        ("it", "Show me the list of Jenkins build failures from yesterday."),
        ("it", "What is the current memory usage on the Redis cluster?"),
        ("it", "Get the OAuth token expiration settings for the API."),
        ("sales", "Show me the win-loss analysis for the last quarter."),
        ("sales", "List all prospects currently in the discovery phase."),
        ("sales", "Display the sales forecast for the upcoming holiday season."),
        ("finance", "Show me the escrow account balance for the merger."),
        ("finance", "List all invoices past 60 days overdue."),
        ("finance", "Retrieve the foreign currency hedging report."),
        ("other", "What is the speed of sound?"),
        ("other", "Give me a recipe for hummus."),
        ("other", "How do you say hello in French?"),
    ]
    for expected, prompt in tests:
        result = predict_domain(prompt)
        print(f"{expected}: {prompt!r} -> {result}")