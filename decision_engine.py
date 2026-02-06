from typing import List, Dict

FULL_SCOPE_PHRASES = [
    "all info",
    "all information",
    "full info",
    "full information",
    "everything",
    "entire profile",
    "complete record",
]


POLICIES = [
    {
        "id": "no_doctor_contact",
        "applies_to_entity": "doctor",
        "applies_to_domains": ["healthcare"],
        "blocked_fields": ["address", "phone"],
        "action": "rewrite",
        "reason": "Doctor contact details must not be exposed"
    },
    {
        "id": "no_efn",
        "applies_to_entity": "employee",
        "applies_to_domains": ["healthcare"],
        "blocked_fields": ["EFN"],
        "action": "deny",
        "reason": "EFN is strictly confidential"
    },
    {
        "id": "no_employee_contact",
        "applies_to_entity": "employee",
        "applies_to_domains": ["*"],
        "allowed_domains": ["hr"],
        "blocked_fields": ["phone", "email"],
        "action": "deny",
        "reason": "Employee contact details must not be exposed"
    },
    {
        "id": "no_user_contact",
        "applies_to_entity": "user",
        "applies_to_domains": ["*"],
        "allowed_domains": ["hr"],
        "blocked_fields": ["email", "phone", "address"],
        "action": "deny",
        "reason": "User email address must not be exposed"
    }
]


def detect_requested_scope(text: str) -> str:
    text = text.lower()
    for phrase in FULL_SCOPE_PHRASES:
        if phrase in text:
            return "full"
    return "partial"


def decide(signal: Dict) -> Dict:
    """
    signal = {
        intent,
        entities,
        mentioned_fields,
        implied_fields
    }
    """

    blocked = []
    reasons = []

    # Conservative scope closure:
    # If request scope is "full", assume all sensitive fields are requested
    SENSITIVE_FIELDS = {"email", "phone", "address", "EFN", "SSN"}

    requested_scope = signal.get("requested_scope", "partial")

    if requested_scope == "full":
        requested_fields = set(SENSITIVE_FIELDS)
    else:
        requested_fields = set(signal.get("mentioned_fields", [])) | set(
            signal.get("implied_fields", [])
        )

    for policy in POLICIES:
        if policy["applies_to_entity"] in signal.get("entities", []):
            for field in policy["blocked_fields"]:
                if field in requested_fields:
                    blocked.append(field)
                    reasons.append(policy["reason"])

                    if policy["action"] == "deny":
                        blocked.append(field)
                        reasons.append(policy["reason"])

    if blocked:
        # if any deny policy matched → deny, else rewrite
        action = "deny" if any(
            p["action"] == "deny"
            and p["applies_to_entity"] in signal.get("entities", [])
            and any(f in requested_fields for f in p["blocked_fields"])
            for p in POLICIES
        ) else "rewrite"

        return {
            "action": action,
            "blocked_fields": sorted(set(blocked)),
            "reason": "; ".join(sorted(set(reasons)))
        }

    return {
        "action": "allow",
        "blocked_fields": [],
        "reason": "No policy violations detected"
    }


if __name__ == "__main__":
    test_signal = {
        "intent": "data_retrieval",
        "entities": ["doctor"],
        "mentioned_fields": ["EFN"],
        "implied_fields": ["address", "phone"]
    }

    decision = decide(test_signal)
    print(decision)