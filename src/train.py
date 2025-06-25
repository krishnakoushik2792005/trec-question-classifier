"""
Fine-tune DistilBERT on TREC with TensorFlow/Keras.
Run:  python -m src.train_tf --epochs 3
"""
import argparse, os, tensorflow as tf
from transformers import (TFDistilBertForSequenceClassification,
                          DistilBertTokenizerFast,
                          create_optimizer)
from datasets import load_dataset
from evaluate import load as load_metric
from .data import load_trec, tokenize_dataset
from .utils import set_seed, get_logger

logger = get_logger("train_tf")

def main(args):
    set_seed()
    # 1. load + tokenise ---------------------------------------------
    ds_raw = load_trec(local_path=args.data_dir)
    ds_tok, tokenizer = tokenize_dataset(ds_raw)

    # convert to tf.data.Dataset (built-in helper)
    train_tf = ds_tok["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=True,
        batch_size=args.batch,
    )
    val_tf = ds_tok["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["label"],
        shuffle=False,
        batch_size=args.batch,
    )

    # 2. build model --------------------------------------------------
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=6
    )

    # 3. compile ------------------------------------------------------
    steps_per_epoch = len(train_tf)
    num_train_steps = steps_per_epoch * args.epochs
    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=int(0.1 * num_train_steps),
    )
    model.compile(optimizer=optimizer,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    # 4. train --------------------------------------------------------
    logger.info("Starting TF training …")
    model.fit(train_tf,
              validation_data=val_tf,
              epochs=args.epochs)

    # 5. save ---------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    logger.info(f"Saved TF model to {args.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir",  default="models/distilbert-trec-tf")
    p.add_argument("--epochs",   type=int, default=3)
    p.add_argument("--batch",    type=int, default=16)
    main(p.parse_args())
