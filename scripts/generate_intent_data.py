import json
from itertools import product
from pathlib import Path

# --------------------
# Vocabulary
# --------------------

ENTITIES = [
    "employee",
    "doctor",
    "customer",
    "user",
    "invoice",
    "transaction",
    "patient",
    "appointment",
    "visit",
    "medical record",
    "medical history",
    "medical report",
    "medical note",
    "medical prescription",
    "medical diagnosis",
]

FIELDS = [
    "salary",
    "address",
    "phone number",
    "email",
    "status",
    "role",
    "EFN",
    "SSN",
    "DOB",
    "gender",
    "race",
    "ethnicity",
    "language",
    "insurance",
    "insurance number",
]

METRICS = [
    "count",
    "total",
    "average",
    "growth",
    "change",
    "difference",
    "standard deviation",
    "variance",
    "trend",
    "change rate",
    "growth rate",
    "decrease rate",
    "increase rate",
    "average change",
    "average growth",
    "average change rate",
    "average growth rate",
    "average decrease rate",
    "average increase rate",
    "average standard deviation",
    "average variance",
    "average trend",
]

READ_TEMPLATES = [
    "Show {entity} {field}",
    "Display {entity} {field}",
    "Get {entity} {field}",
    "View {entity} {field}",
    "List {entity} {field}",
    "Give me {entity} {field}",
    "Extract {entity} {field}",
    "Retrieve {entity} {field}",
    "Show me {entity} {field}",
    "Get me {entity} {field}",
]

READ_METRIC_TEMPLATES = [
    "Show {entity} {metric}",
    "Display {metric} of {entity}s",
    "Compare {entity} {metric} over time",
    "Show change in {entity} {metric}",
    "Summarize {entity} {field}",
    "Give me a summary of {entity} {field}",
    "Condense {entity} {field}",
    "Create a short summary of {entity} {field}",
    "Summarize key points from {entity} {field}",
    "TLDR {entity} {field}",
    "Give me bullet points from {entity} {field}",
]

WRITE_TEMPLATES = [
    "Update {entity} {field}",
    "Change {entity} {field}",
    "Set {entity} {field}",
    "Modify {entity} {field}",
    "Remove {field} from {entity}",
    "Delete {entity} record",
    "Add {entity} {field}",
    "Create {entity} {field}",
    "Insert {entity} {field}",
    "Delete {field} from {entity}",
]

# --------------------
# Generation
# --------------------

examples = []

# READ: entity + field
for entity, field, template in product(ENTITIES, FIELDS, READ_TEMPLATES):
    text = template.format(entity=entity, field=field)
    examples.append({"text": text, "label": "read"})

# READ: metrics (templates may use {metric} and/or {field}; use metric for both)
for entity, metric, template in product(ENTITIES, METRICS, READ_METRIC_TEMPLATES):
    text = template.format(entity=entity, metric=metric, field=metric)
    examples.append({"text": text, "label": "read"})

# WRITE: entity + field
for entity, field, template in product(ENTITIES, FIELDS, WRITE_TEMPLATES):
    text = template.format(entity=entity, field=field)
    examples.append({"text": text, "label": "write"})

# --------------------
# Output
# --------------------

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "intent_train.jsonl"
LABEL_ORDER = {"other": 0, "read": 1, "write": 2}

# Load existing examples and track existing texts
existing_examples = []
existing_texts = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            existing_examples.append(ex)
            existing_texts.add(ex["text"])

# Add only generated examples whose text is not already present
new_count = 0
for ex in examples:
    if ex["text"] not in existing_texts:
        existing_examples.append(ex)
        existing_texts.add(ex["text"])
        new_count += 1

# Sort: other first, read in between, write at the end
existing_examples.sort(key=lambda ex: (LABEL_ORDER.get(ex["label"], 3), ex["text"]))

with open(OUTPUT_PATH, "w") as f:
    for ex in existing_examples:
        f.write(json.dumps(ex) + "\n")

print(f"Generated {len(examples)} intent examples ({new_count} new, {len(existing_examples) - new_count} existing)")
print(f"Wrote {len(existing_examples)} total examples to {OUTPUT_PATH}")