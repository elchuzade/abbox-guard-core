from typing import Dict

# Intent analyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Local modules
from field_extractor import extract_fields_and_entities
from decision_engine import decide
from prompt_rewriter import rewrite_prompt

GRANULARITY_CONFIDENCE_THRESHOLD = 0.75

INTENT_MODEL_DIR = "intent_model"

intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_DIR)
intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_DIR)
intent_model.eval()

DOMAIN_MODEL_DIR = "domain_model"

domain_tokenizer = AutoTokenizer.from_pretrained(DOMAIN_MODEL_DIR)
domain_model = AutoModelForSequenceClassification.from_pretrained(DOMAIN_MODEL_DIR)
domain_model.eval()

GRANULARITY_MODEL_DIR = "granularity_model"

granularity_tokenizer = AutoTokenizer.from_pretrained(GRANULARITY_MODEL_DIR)
granularity_model = AutoModelForSequenceClassification.from_pretrained(GRANULARITY_MODEL_DIR)
granularity_model.eval()

def predict_intent(prompt: str) -> Dict:
    inputs = intent_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits = intent_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    intent = intent_model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    return {
        "intent": intent,
        "confidence": round(confidence, 3)
    }


def predict_domain(prompt: str) -> dict:
    inputs = domain_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = domain_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    domain = domain_model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    return {"domain": domain, "confidence": round(confidence, 3)}


def predict_granularity(prompt: str) -> dict:
    inputs = granularity_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = granularity_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    granularity = granularity_model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    return {"granularity": granularity, "confidence": round(confidence, 3)}


def run_guardrail(prompt: str) -> Dict:
    # Intent analysis (ML)
    intent_result = predict_intent(prompt)
    
    # Domain analysis (ML)
    domain_result = predict_domain(prompt)

    # Field & entity extraction (rules)
    extraction = extract_fields_and_entities(prompt)

    # Granularity analysis (ML)
    granularity_result = predict_granularity(prompt)

    # Conservative granularity override:
    # Never allow aggregate when confidence is low or scope is full
    granularity = granularity_result["granularity"]
    granularity_conf = granularity_result["confidence"]
    if (
        granularity_conf < GRANULARITY_CONFIDENCE_THRESHOLD
        or extraction.get("requested_scope") == "full"
    ):
        granularity = "record_level"

    # Build signal object for decision engine
    signal = {
        "intent": intent_result["intent"],
        "domain": domain_result["domain"],
        "entities": extraction["entities"],
        "mentioned_fields": extraction["mentioned_fields"],
        "implied_fields": extraction["implied_fields"],
        "requested_scope": extraction.get("requested_scope", "partial"),
        "granularity": granularity,
    }

    # Decision
    decision = decide(signal)

    # Rewrite if needed
    final_prompt = prompt
    if decision["action"] == "rewrite":
        final_prompt = rewrite_prompt(prompt, decision)

    return {
        "original_prompt": prompt,
        "intent": intent_result["intent"],
        "intent_confidence": intent_result["confidence"],
        "domain": domain_result["domain"],
        "domain_confidence": domain_result["confidence"],
        "granularity": granularity_result["granularity"],
        "granularity_confidence": granularity_result["confidence"],
        "entities": extraction["entities"],
        "mentioned_fields": extraction["mentioned_fields"],
        "implied_fields": extraction["implied_fields"],
        "requested_scope": extraction.get("requested_scope", "partial"),
        "decision": decision,
        "suggested_alternatives": decision.get("suggested_alternatives", []),
        "final_prompt": final_prompt
    }


if __name__ == "__main__":
    prompt = "Show me 1 doctor with all information"
    result = run_guardrail(prompt)

    from pprint import pprint
    pprint(result)