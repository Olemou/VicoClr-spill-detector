import numpy as np
from pathlib import Path
from importlib.resources import files
from benchmarking.utils import load_setup, load_models_clr
from src_utils.logging import get_logger
from src.model import VisionTransformer, load_checkpoint
from src.vit_classifier import ViTClassifier
from src.ddp import is_main_process


MODES = [
    "full",
    "no_uncertainty",
    "uncertainity_curriculum_lr",
    "no_weighting",
    "uncertainty_only",
]



logger = get_logger(__name__)


def load_models_classifier(args,modes=MODES):
   
    if not is_main_process():
        return {}

   
    vit_models, _ = load_models_clr(MODES)
    
    checkpoint_folder = files("checkpoint.classifier")
    

    models = {}

    for mode in modes:

        checkpoint_path = (
            checkpoint_folder
            / mode
            / f"checkpoint_r{args.rank}.pth"
        )
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.warning(
                f"[WARNING] Checkpoint not found for mode "
                f"'{mode}': {checkpoint_path}"
            )
            continue

        try:
            model = ViTClassifier(
                vit_model=vit_models[mode],
                model_size=args.model_size,
                
            )

            _ = load_checkpoint(
                checkpoint_path,
                model,
            )
            models[mode] = model.to(args.device)

            logger.info(
                f"[INFO] Loaded model for mode "
                f"'{mode}'"
            )

        except Exception as e:
            logger.error    (
                f"[ERROR] Failed loading mode "
                f"'{mode}': {e}"
            )

    return models, args.device
    