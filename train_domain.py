import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "distilroberta-base"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def main():
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/domain_train.jsonl",
            "validation": "data/domain_valid.jsonl",
        },
    )

    labels = sorted(set(dataset["train"]["label"]))
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    def encode_labels(example):
        example["label_id"] = label2id[example["label"]]
        return example

    dataset = dataset.map(encode_labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text", "label"])
    tokenized = tokenized.rename_column("label_id", "labels")
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    args = TrainingArguments(
        output_dir="out_domain",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        eval_strategy="epoch",     # transformers v5
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("domain_model")
    tokenizer.save_pretrained("domain_model")

    print("Saved to ./domain_model")
    print("Labels:", labels)

if __name__ == "__main__":
    main()