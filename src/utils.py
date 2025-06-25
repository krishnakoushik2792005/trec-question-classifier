"""
Shared helpers: seed fixing, simple colored logger, confusion-matrix plot.
"""

import random, os, numpy as np, torch, logging, matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(name: str = "trec") -> logging.Logger:
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logging.basicConfig(format=fmt, datefmt="%H:%M:%S", level=logging.INFO)
    return logging.getLogger(name)


def save_confusion_matrix(y_true, y_pred, labels, out_path: str) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", xticks_rotation=35
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
