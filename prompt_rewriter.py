from typing import List, Dict

# Canonical phrasing for blocked fields
FIELD_REWRITE_MAP = {
    "address": "addresses",
    "phone": "contact details",
    "email": "email addresses",
    "EFN": "internal identifiers",
    "ssn": "government identifiers",
}


def humanize_fields(fields: List[str]) -> str:
    phrases = []
    for f in fields:
        phrases.append(FIELD_REWRITE_MAP.get(f, f))
    return ", ".join(sorted(set(phrases)))


def rewrite_prompt(original_prompt: str, decision: Dict) -> str:
    """
    decision = {
        action: "rewrite",
        blocked_fields: [...],
        reason: "..."
    }
    """

    if decision["action"] != "rewrite":
        return original_prompt

    blocked_fields = decision.get("blocked_fields", [])
    if not blocked_fields:
        return original_prompt

    exclusions = humanize_fields(blocked_fields)

    # Remove risky phrases
    rewritten = original_prompt
    rewritten = rewritten.replace("all info", "professional information")
    rewritten = rewritten.replace("all information", "professional information")

    # Append explicit exclusion clause
    rewritten = f"{rewritten}, excluding {exclusions}."

    return rewritten


if __name__ == "__main__":
    original_prompt = "Show me 3 doctors with all info except EFN"

    decision = {
        "action": "rewrite",
        "blocked_fields": ["address", "phone"],
        "reason": "Doctor contact details must not be exposed"
    }

    rewritten = rewrite_prompt(original_prompt, decision)
    print(rewritten)