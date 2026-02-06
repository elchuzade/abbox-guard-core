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


SAFE_AGGREGATE_TEMPLATES = {
    "employee": [
        "Show employee count by department",
        "Show average salary by department",
        "Show headcount trend over time"
    ],
    "customer": [
        "Show customer count by region",
        "Show customers grouped by plan",
        "Show average customer lifetime value"
    ],
    "doctor": [
        "Show number of doctors by specialty",
        "Show doctor count by clinic",
        "Show appointments per specialty"
    ]
}


POLICIES = [
    {
        "id": "no_doctor_contact_record_level",
        "applies_to_entity": "doctor",
        "applies_to_domains": ["healthcare"],
        "blocked_fields": ["address", "phone"],
        "blocked_granularity": ["record_level"],
        "action": "rewrite",
        "reason": "Doctor contact details must not be exposed at record level"
    },
    {
        "id": "no_employee_efn_any_level",
        "applies_to_entity": "employee",
        "applies_to_domains": ["*"],
        "blocked_fields": ["EFN"],
        "blocked_granularity": ["record_level", "aggregate"],
        "action": "deny",
        "reason": "EFN is strictly confidential"
    },
    {
        "id": "employee_contact_hr_only",
        "applies_to_entity": "employee",
        "applies_to_domains": ["*"],
        "allowed_domains": ["hr"],
        "blocked_fields": ["phone", "email"],
        "blocked_granularity": ["record_level"],
        "action": "deny",
        "reason": "Employee contact details may only be accessed by HR at record level"
    },
    {
        "id": "user_contact_hr_only",
        "applies_to_entity": "user",
        "applies_to_domains": ["*"],
        "allowed_domains": ["hr"],
        "blocked_fields": ["email", "phone", "address"],
        "blocked_granularity": ["record_level"],
        "action": "deny",
        "reason": "User contact details may only be accessed by HR at record level"
    },
    {
        "id": "salary_aggregate_only",
        "applies_to_entity": "employee",
        "applies_to_domains": ["*"],
        "blocked_fields": ["salary"],
        "blocked_granularity": ["record_level"],
        "allowed_granularity": ["aggregate"],
        "action": "deny",
        "reason": "Salary data is only accessible in aggregate form"
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
    granularity = signal.get("granularity", "record_level")

    if requested_scope == "full":
        requested_fields = set(SENSITIVE_FIELDS)
    else:
        requested_fields = set(signal.get("mentioned_fields", [])) | set(
            signal.get("implied_fields", [])
        )

    for policy in POLICIES:
        if policy["applies_to_entity"] not in signal.get("entities", []):
            continue

        # Domain allow-list check
        if policy.get("allowed_domains"):
            if signal.get("domain") not in policy["allowed_domains"]:
                pass
            else:
                continue

        # Granularity allow-list check
        if policy.get("allowed_granularity"):
            if granularity in policy["allowed_granularity"]:
                continue

        # Granularity block check
        if policy.get("blocked_granularity"):
            if granularity not in policy["blocked_granularity"]:
                continue

        for field in policy["blocked_fields"]:
            if field in requested_fields:
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

        decision = {
            "action": action,
            "blocked_fields": sorted(set(blocked)),
            "reason": "; ".join(sorted(set(reasons)))
        }

        # Deterministic safe alternatives:
        # Only suggest when record-level data is blocked
        if granularity == "record_level":
            entities = signal.get("entities", [])
            if entities:
                entity = entities[0]
                decision["suggested_alternatives"] = SAFE_AGGREGATE_TEMPLATES.get(entity, [])

        return decision

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