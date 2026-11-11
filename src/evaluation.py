"""Spoločné pomocné funkcie na vyhodnotenie modelov."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    """Accuracy, Precision, Recall a F1 pre pozitívnu triedu (1)."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def save_results(path, rows):
    """Uloží riadky s metrikami do CSV.

    Ak súbor už existuje, riadky s rovnakou dvojicou (experiment, split)
    sa prepíšu a ostatné ostanú zachované. Preto sa dá notebook spúšťať
    opakovane a neskôr sa dajú pridať ďalšie experimenty.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(new["experiment"], new["split"]))
        keep = [(e, s) not in keys for e, s in zip(old["experiment"], old["split"])]
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def plot_confusion(y_true, y_pred, title, labels=("human", "bot"), save_path=None):
    """Nakreslí confusion matrix a voliteľne ju uloží ako obrázok."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=list(labels), ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
