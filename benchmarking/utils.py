from numpy.random import shuffle
from src.ddp import (
    init_distributed_mode,
    synchronize,
    setup_environment,
    seed_everything,
    parse_arguments,
)
from pathlib import Path
from importlib.resources import files
from src.ddp import is_main_process
from src_utils.logging import get_logger
from src.data_manager import test_init_data
from src.utils import DataIterator
import torch
from src.model import VisionTransformer, load_checkpoint
from typing import Tuple, Optional
logger = get_logger(__name__)
import warnings
warnings.filterwarnings("ignore", message="NumExpr defaulting to")
def load_setup():
    # ------------------------------------------------------
    # DDP Setup
    # ------------------------------------------------------
     # Setup environment and arguments
    setup_environment()
    args = parse_arguments()
    args = init_distributed_mode(args)
    seed_everything(args.seed)
    
    if args.distributed:
        synchronize()
    return args

def get_dataloader(args) -> Tuple[Optional[DataIterator], torch.utils.data.DataLoader, 
                               torch.utils.data.DataLoader]:
    """
    Initialize test dataloaders for contrastive (CLR) and classification evaluation.
    """
    
    test_clr: Optional[DataIterator] = None
    test_classifier_loader: Optional[torch.utils.data.DataLoader] = None

    # Common parameters
    eval_batch_size = min(32, args.batch_size)
    eval_num_workers = min(4, args.num_workers)

    try:
        # ==================== Contrastive Loader (CLR) ====================
        
        test_loader_clr = test_init_data(
            test_data_dir=args.test_data_dir,
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
            drop_last=False,
            shuffle = True,
            rank=args.rank,
            isContrastive=True,
            world_size=args.world_size,
        )

       
        # ==================== Classification Loader ====================
        test_loader_classifier = test_init_data(
            test_data_dir=args.test_data_dir,
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
            drop_last=False,
            rank=args.rank,
            shuffle = True,
            isContrastive=False,
            world_size=args.world_size,
        )



        logger.info(
            f"Evaluation dataloaders created successfully. "
            f"Batch size: {eval_batch_size} | Workers: {eval_num_workers}"
        )

    except Exception as e:
        logger.error(f"Failed to create evaluation dataloaders: {e}", exc_info=True)
        raise

    return test_loader_clr, test_loader_classifier



def load_models_clr(modes:dict):

    args = load_setup()

    if not is_main_process():
        return {}, args.device

    checkpoint_folder = files("checkpoint")

    models = {}

    for mode in modes:

        checkpoint_path = (
            checkpoint_folder / "contrastive"
            / mode
            / f"checkpoint_r{args.rank}.pth"
        )
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint not found for {mode}: "
                f"{checkpoint_path}"
            )
            continue

        try:
            model = VisionTransformer(
                model_size=args.model_size,
                patch_dropout_prob=0.2,
            )

            load_checkpoint(
                checkpoint_path,
                model,
            )

            model = model.to(args.device)
            model.eval()

            models[mode] = model

            logger.info(
                f"Loaded model: {mode}"
            )

        except Exception as e:
            logger.error(
                f"Failed loading {mode}: {e}"
            )

    return models, args.device