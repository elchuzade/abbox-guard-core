# abbox-guard-core

**abbox-guard-core** is the open-source intelligence layer behind **Abbox Guard** — a policy-aware guardrail engine for AI and data-access pipelines.

It analyzes user prompts, extracts intent and domain signals, applies deterministic policy enforcement, and produces safe allow / deny / rewrite decisions.

This repository contains the **core reasoning and ML signal extraction**, not the hosted API or enterprise enforcement layer.

---

## What this repository does

Given:

- a natural-language user prompt
- structured policies written by a compliance manager

It produces:

- intent classification (read vs write, etc.)
- domain classification (HR, healthcare, finance, …)
- extracted entities and fields
- conservative safety decisions:
  - allow
  - deny
  - rewrite (with blocked fields)

---

## What this repository does NOT do

- ❌ Execute SQL or mutate data
- ❌ Act as an authorization system
- ❌ Store customer data
- ❌ Replace enterprise access control

It is a **guardrail**, not an executor.

---

## Repository structure

```
abbox-guard-core/
├── orchestrator.py          # End-to-end pipeline runner
├── decision_engine.py       # Deterministic policy evaluation
├── field_extractor.py       # Entity, field, scope extraction
│
├── train_intent.py          # Intent classifier training
├── train_domain.py          # Domain classifier training
│
├── intent_model/            # Trained intent model artifacts
├── domain_model/            # Trained domain model artifacts
│
├── data/
│   ├── train.jsonl          # Intent training data
│   ├── valid.jsonl
│   ├── domain_train.jsonl   # Domain training data
│   └── domain_valid.jsonl
│
├── tests/                   # (optional) tests
└── README.md
```

---

## Core pipeline

```
Prompt
  ↓
Intent classifier (ML)
  ↓
Domain classifier (ML)
  ↓
Field & entity extraction (rules)
  ↓
Conservative scope closure
  ↓
Policy decision engine (deterministic)
  ↓
Allow / Deny / Rewrite decision
```

Only **intent and domain classification** use machine learning.  
All enforcement is deterministic and auditable.

---

## Running the system locally

### 1. Set up environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train models (optional if already trained)

```bash
python3 train_intent.py
python3 train_domain.py
```

### 3. Run the orchestrator

```bash
python3 orchestrator.py
```

You will see a structured decision output for a sample prompt.

---

## Policies

Policies are defined as structured JSON-like objects.

Example:

```json
{
  "id": "employee_contact_access",
  "applies_to_entity": "employee",
  "blocked_fields": ["address", "phone"],
  "allowed_domains": ["hr"],
  "action": "deny",
  "reason": "Employee contact details may only be accessed by HR"
}
```

### Policy principles

- Default deny for sensitive fields
- Allow-lists preferred over exclude-lists
- Conservative interpretation of vague prompts
- ML signals never directly allow execution

---

## Open-source vs commercial

**Open-source (this repo):**

- intent & domain models
- extraction logic
- rewrite logic
- pipeline orchestration

**Commercial (Abbox Guard SaaS):**

- policy ingestion from free text
- enterprise enforcement rules
- audit logs & approvals
- production API & scaling

---

## Contributing

Contributions are welcome for:

- datasets
- model improvements
- new domains
- better rewrite logic
- tests & documentation

Please do not submit changes that alter enforcement semantics without discussion.

---

## License

Apache 2.0 (recommended) or MIT  
Final license to be confirmed.

---

## About Abbox

**Abbox** builds policy-aware safety infrastructure for AI systems.
