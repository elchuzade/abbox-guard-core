import re
from typing import List, Dict

# 1) Known entities and how users refer to them
ENTITY_SYNONYMS = {
    "doctor": ["doctor", "doctors", "physician", "medical staff"],
    "employee": ["employee", "employees", "staff", "worker"],
    "user": ["user", "users", "account holder"],
    "customer": ["customer", "customers", "client"]
}

# 2) Known sensitive fields and their synonyms
FIELD_SYNONYMS = {
    "EFN": ["efn", "employee file number"],
    "phone": ["phone", "phone number", "contact number", "mobile"],
    "address": ["address", "home address", "location"],
    "email": ["email", "email address"],
    "ssn": ["ssn", "social security number"]
}

# 3) Fields that are usually implied when someone says "all info"
IMPLIED_FIELDS_BY_ENTITY = {
    "doctor": ["address", "phone"],
    "employee": ["address", "phone", "email"],
    "user": ["email"]
}

FULL_SCOPE_PHRASES = [
    "all info",
    "all information",
    "full info",
    "full information",
    "everything",
    "entire profile",
    "complete record",
]

def detect_requested_scope(text: str) -> str:
    text = text.lower()
    for phrase in FULL_SCOPE_PHRASES:
        if phrase in text:
            return "full"
    return "partial"


def normalize(text: str) -> str:
    return text.lower()


def extract_entities(text: str) -> List[str]:
    text = normalize(text)
    found = []

    for entity, synonyms in ENTITY_SYNONYMS.items():
        for s in synonyms:
            if re.search(rf"\b{s}\b", text):
                found.append(entity)
                break

    return found


def extract_mentioned_fields(text: str) -> List[str]:
    text = normalize(text)
    found = []

    for field, synonyms in FIELD_SYNONYMS.items():
        for s in synonyms:
            if re.search(rf"\b{s}\b", text):
                found.append(field)
                break

    return found


def extract_implied_fields(text: str, entities: List[str]) -> List[str]:
    text = normalize(text)
    implied = []

    if "all info" in text or "all information" in text:
        for entity in entities:
            implied.extend(IMPLIED_FIELDS_BY_ENTITY.get(entity, []))

    return list(set(implied))


def extract_fields_and_entities(text: str) -> Dict:
    entities = extract_entities(text)
    mentioned_fields = extract_mentioned_fields(text)
    implied_fields = extract_implied_fields(text, entities)
    requested_scope = detect_requested_scope(text)

    implied_fields = [
        f for f in implied_fields if f not in mentioned_fields
    ]

    return {
        "entities": entities,
        "mentioned_fields": mentioned_fields,
        "implied_fields": implied_fields,
        "requested_scope": requested_scope
    }


if __name__ == "__main__":
    prompt = "Show me 3 doctors with all info except EFN"
    result = extract_fields_and_entities(prompt)
    print(result)