import shutil
from importlib.resources import files
from pathlib import Path
from src_utils.logging import get_logger
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from benchmarking.eval_model import load_models_clr
from benchmarking.utils import get_dataloader

logger = get_logger(__name__)
MODES = [
    "full",
    "no_uncertainty",
    "uncertainity_curriculum_lr",
    "no_weighting",
    "uncertainty_only",
]


def generate_tsne_data(
    model,
    data_iter,
    device,
    mode: str,
):
    checkpoint_folder_clr = files("benchmarking") / "tsne_output"

    output_dir = (
        Path(checkpoint_folder_clr)
        / f"tsne_{mode}"
    )

    # Remove previous results
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = []
    labels_list = []

    model.eval()

    with torch.no_grad():
        for batch in data_iter:

            inputs = batch["images"]
            labels = batch["labels"]

            x1 = inputs[:, :3]
            x2 = inputs[:, 3:]

            x1 = x1.to(
                device,
                non_blocking=True,
            )
            x2 = x2.to(
                device,
                non_blocking=True,
            )

            images = torch.cat(
                [x1, x2],
                dim=0,
            )

            labels_concat = labels.repeat(2)

            feats = model(images)

            # ViT output: [B, Tokens, D]
            if feats.dim() == 3:
                feats = feats.mean(dim=1)

            features.append(
                feats.cpu()
            )

            labels_list.append(
                labels_concat.cpu()
            )

    features = torch.cat(
        features,
        dim=0,
    ).numpy()

    labels = torch.cat(
        labels_list,
        dim=0,
    ).numpy()


    logger.info(
                 f"[{mode}] Features shape: {features.shape}"
            )

    # --------------------------------------------------
    # Normalize features
    # --------------------------------------------------
    scaler = StandardScaler()

    features_norm = scaler.fit_transform(
        features
    )

    # --------------------------------------------------
    # Adaptive perplexity
    # --------------------------------------------------
    n_samples = len(features_norm)

    perplexity = min(
        50,
        max(
            5,
            (n_samples - 1) // 3,
        ),
    )

    perplexity = min(
        perplexity,
        n_samples - 1,
    )

    print(
        f"[{mode}] Running t-SNE with "
        f"{n_samples} samples "
        f"(perplexity={perplexity})"
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    features_2d = tsne.fit_transform(
        features_norm
    )

    # --------------------------------------------------
    # Save Torch
    # --------------------------------------------------
    torch.save(
        {
            "features": torch.from_numpy(
                features
            ),
            "labels": torch.from_numpy(
                labels
            ),
            "features_2d": torch.from_numpy(
                features_2d
            ),
        },
        output_dir / f"tsne_features_{mode}.pt"
    )

    # --------------------------------------------------
    # Save NumPy
    # --------------------------------------------------
    np.savez(
        output_dir / "tsne_data.npz",
        features=features,
        labels=labels,
        features_2d=features_2d,
    )

    # --------------------------------------------------
    # Save DataFrame
    # --------------------------------------------------
    df = pd.DataFrame(
        {
            "tSNE_1": features_2d[:, 0],
            "tSNE_2": features_2d[:, 1],
            "label": labels.astype(str),
        }
    )

    df.to_csv(
        output_dir / "tsne_dataframe.csv",
        index=False,
    )

    df.to_pickle(
        output_dir / "tsne_dataframe.pkl"
    )

    logger.info(
        f"[{mode}] Saved to: {output_dir}"
    )

    return df


def load_tsne_dataframe(
    mode: str,
    use_pickle: bool = True,
):
    checkpoint_folder_clr = files("benchmarking") / "tsne_output"

    output_dir = (
        Path(checkpoint_folder_clr)
        / f"tsne_{mode}"
    )

    if use_pickle:
        return pd.read_pickle(
            output_dir / "tsne_dataframe.pkl"
        )

    return pd.read_csv(
        output_dir / "tsne_dataframe.csv"
    )


def launch_tsne_data_generation(args):

    models, device = load_models_clr(MODES)

    if len(models) == 0:
        print("No models loaded.")
        return

    test_loader, _ = get_dataloader(args)

    for mode, model in models.items():
      logger.info(
            f"\n{'='*60}\n"
            f"Generating t-SNE for {mode}\n"
            f"{'='*60}"
        )
      generate_tsne_data(
            model=model,
            data_iter=test_loader,
            device=device,
            mode=mode,
        )
