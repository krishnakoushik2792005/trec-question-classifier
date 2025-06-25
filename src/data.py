"""
Dataset download / local load + tokenisation utilities.
"""

from typing import Tuple
from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizerFast
from .utils import get_logger

logger = get_logger("data")


def load_trec(local_path: str | None = None) -> DatasetDict:
    """
    • local_path: load from train_5500.label / test.label (string labels).
    • else: load SetFit/TREC-QC mirror (has both fine & coarse numeric labels).
    Returns DatasetDict with columns ['text','label'] where label ∈ [0..5].
    """
    if local_path:
        # —— Local .label files branch —————————————————————————————
        data_files = {
            "train": f"{local_path}/train_5500.label",
            "test":  f"{local_path}/test.label",
        }
        ds = load_dataset("text", data_files=data_files)

        def split_example(e):
            head, *rest = e["text"].split()
            e["label_str"] = head.split(":")[0]   # e.g. "LOC"
            e["text"] = " ".join(rest)
            return e

        ds = ds.map(split_example)
        lbl2id = {k: i for i, k in enumerate(
            ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
        )}
        ds = ds.map(lambda e: {"label": lbl2id[e["label_str"]]})
        ds = ds.remove_columns("label_str")

    else:
        # —— Remote mirror branch ————————————————————————————————
        ds = load_dataset("SetFit/TREC-QC")

        # 1️⃣ Drop any existing 'label' (fine-grained) column
        if "label" in ds["train"].column_names:
            ds = ds.remove_columns("label")

        # 2️⃣ Drop all text/original variants
        for col in [
            "label_text", "label_original",
            "label_fine", "label-fine",
            "label_coarse_text", "label_coarse_original"
        ]:
            if col in ds["train"].column_names:
                ds = ds.remove_columns(col)

        # 3️⃣ Rename the true numeric coarse column → 'label'
        if "label_coarse" in ds["train"].column_names:
            ds = ds.rename_column("label_coarse", "label")
        else:
            raise ValueError("`label_coarse` not found in SetFit/TREC-QC")

    logger.info(
        f"Loaded TREC – {len(ds['train'])} train / {len(ds['test'])} test samples."
    )
    return ds


def tokenize_dataset(
    ds: DatasetDict,
    max_length: int = 64,
) -> Tuple[DatasetDict, DistilBertTokenizerFast]:
    """
    Tokenises the 'text' field and sets output for PyTorch:
      columns=['input_ids','attention_mask','label']
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )

    ds_tok = ds.map(
        lambda batch: tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )
    ds_tok.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )
    logger.info("Tokenisation complete.")
    return ds_tok, tokenizer
