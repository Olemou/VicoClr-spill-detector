from benchmarking.eval_model import load_models_classifier
from benchmarking.utils import get_dataloader
import shutil
from pathlib import Path
from importlib.resources import files
from src_utils.logging import get_logger
logger = get_logger(__name__)

import torch
import numpy as np
import pandas as pd
import os
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# -----------------------------
# SINGLE EVALUATION
# -----------------------------


def evaluate_classifier(args,model, device):
    _, test_loader = get_dataloader(args)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():

        for batch in test_loader:
           
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
           

            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
    "accuracy": accuracy_score(y_true, y_pred) * 100,
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
    "precision": precision_score(y_true, y_pred, zero_division=0, average="macro") * 100,
    "recall": recall_score(y_true, y_pred, zero_division=0, average="macro") * 100,
    "f1": f1_score(y_true, y_pred, zero_division=0, average="macro") * 100,
}

    cm = confusion_matrix(y_true, y_pred)

    return metrics, cm


# -----------------------------
# 10 RUNS PER MODEL
# -----------------------------
def evaluate_model_3_times(args,model, device, n_runs=10):

    all_metrics = []
    last_cm = None

    for run in range(n_runs):

        metrics, cm = evaluate_classifier(
          args,
            model=model,
            device=device,
        )

        all_metrics.append(metrics)

    
        last_cm = cm

    df = pd.DataFrame(all_metrics)

    summary = {}

    for k in df.columns:
        mean = df[k].mean()
        std = df[k].std(ddof=1)

        summary[f"{k}_mean"] = mean
        summary[f"{k}_std"] = std
        summary[f"{k}_paper"] = f"{mean:.2f} ± {std:.2f}"

    return df, summary, last_cm


# -----------------------------
# SAVE RESULTS
# -----------------------------
def save_results(results_df, confusion_matrices):
    
    checkpoint_folder = files("benchmarking")
    output_dir = checkpoint_folder / "classifier_results"
    output_dir =Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
   

    # metrics table
    results_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # confusion matrices
    for model_name, cm in confusion_matrices.items():
        np.save(
            os.path.join(output_dir, f"{model_name}_cm.npy"),
            cm
        )

    # summary json
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=4)


# -----------------------------
# ALL MODELS EVALUATION
# -----------------------------
def evaluate_all_models(args,n_runs:int = 3, isSave:bool=True):

    results = []
    confusion_matrices = {}
    models, device = load_models_classifier(args)

    for mode, model in models.items():

        logger.info(f"Evaluating {mode}...")

        _, summary, cm_last = evaluate_model_3_times(
          args =args,
            model=model,
            device=device,
            n_runs=n_runs,
        )

        row = {
            "mode": mode,

            "accuracy": summary["accuracy_mean"],
            "accuracy_std": summary["accuracy_std"],

            "precision": summary["precision_mean"],
            "precision_std": summary["precision_std"],

            "recall": summary["recall_mean"],
            "recall_std": summary["recall_std"],

            "f1": summary["f1_mean"],
            "f1_std": summary["f1_std"],

            "balanced_accuracy": summary["balanced_accuracy_mean"],
            "balanced_accuracy_std": summary["balanced_accuracy_std"],
        }

        results.append(row)

        # store ONLY last run CM
        confusion_matrices[mode] = cm_last

    results_df = pd.DataFrame(results)

    # save if needed
    if isSave:
        save_results(results_df, confusion_matrices)

    return results_df, confusion_matrices