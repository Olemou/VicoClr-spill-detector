from src.classifier_training import train_classifier
from src.training_contrastive_lr import train_contrastive
from src.ddp import init_distributed_mode, is_main_process,setup_environment, seed_everything,parse_arguments
from src.utils import get_logger
logger = get_logger(__name__)

def main():
    # Train contrastive model
    args = parse_arguments()
    setup_environment()
    init_distributed_mode(args)
    seed_everything(args.seed)
    logger.info("Starting contrastive training...")
    train_contrastive(args)
    
    # Train classifier model
    logger.info("Starting classifier training...")
    train_classifier(args)
if __name__ == "__main__":
    main()
    
 
