"""
Evaluate saved TF checkpoint + make confusion matrix.
Run: python -m src.eval_tf --ckpt models/distilbert-trec-tf
"""
import os
import argparse
import numpy as np
from transformers import TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report
from src.data import load_trec, tokenize_dataset
from src.utils import save_confusion_matrix, get_logger

logger = get_logger("eval_tf")

LABELS = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]


def main(args):
    # Ensure reports directory exists relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Load dataset and tokenize
    ds_raw = load_trec()
    ds_tok, _ = tokenize_dataset(ds_raw)

    # Load the TF model
    model = TFDistilBertForSequenceClassification.from_pretrained(args.ckpt)
    preds, gold = [], []

    # Iterate over test examples
    for batch in ds_tok["test"]:
        # Convert torch tensors to NumPy for TF model
        input_ids = batch["input_ids"].numpy()[None, :]
        attention_mask = batch["attention_mask"].numpy()[None, :]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds.append(int(np.argmax(logits, axis=-1)))
        gold.append(int(batch["label"]))

    # Print classification metrics
    print(classification_report(gold, preds, target_names=LABELS))

    # Save confusion matrix
    out_path = os.path.join(reports_dir, "metrics_tf.png")
    save_confusion_matrix(gold, preds, LABELS, out_path)
    logger.info(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/distilbert-trec-tf")
    main(p.parse_args())
