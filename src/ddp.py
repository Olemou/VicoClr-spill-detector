
import os
import sys
import argparse
import socket
import os
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
import random
from src_utils.logging import get_logger
logger = get_logger("DDP Training and Evaluation setup", force=True)
import warnings
warnings.filterwarnings("error")  
warnings.filterwarnings("ignore", category=DeprecationWarning)

#=============================================================================
# -------------------------------
# Helper to set requires_grad
# -------------------------------
def set_trainable(module, flag: bool):
    """Set requires_grad for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = flag

# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Configure environment variables for multi-node training."""
    
    # Optimize CPU threading
   
    # PyTorch optimizations
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # Prevent tokenizer parallelism warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility across all nodes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments for multi-node training."""
    
    parser = argparse.ArgumentParser(
        description="Multi-Node Distributed Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Distributed arguments (usually set by torchrun)
    parser.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", 1)),
                        help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=int(os.environ.get("NODE_RANK", 0)),
                        help="Rank of current node")
    parser.add_argument("--nproc_per_node", type=int, 
                        default=int(os.environ.get("NPROC_PER_NODE", torch.cuda.device_count())),
                        help="Number of processes per node")
    parser.add_argument("--master_addr", type=str, 
                        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
                        help="Master node IP address")
    parser.add_argument("--master_port", type=int, 
                        default=int(os.environ.get("MASTER_PORT", 29500)),
                        help="Master node port")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Data loading workers per GPU")
    parser.add_argument("--loss_reg_std_mult", type=float, default=3.0,
                        help="Standard deviation multiplier for loss regularization")
    parser.add_argument("--loss_reg_min_epoch", type=int, default=5,
                        help="Minimum epoch to start loss regularization")  
    parser.add_argument("--loss_reg_num_tracking_steps", type=int, default=300,
                        help="Number of steps to track for loss regularization")
    parser.add_argument("--save_every_freq", type=int, default=5,
                        help="Frequency (in epochs) to save checkpoints (0 to disable)")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of warmup epochs")
    parser.add_argument("--numbre_epoch_classifier",type=int, default=50) 

    # 

    parser.add_argument("--compute_loss_mode", type=str, default="full")
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--test_data_dir", type=str,default="/scratch/vico_clr/test_data")
    parser.add_argument("--root_dir", type=str, default="/scratch/vico_clr/data")
    parser.add_argument("--continuing_training",action="store_true") 
    parser.add_argument("--lr_classifier", type=float, default=1e-4,
                        help="Learning rate for classifier training") 
    parser.add_argument("--weight_decay_classifier", type=float, default=0.009,
                        help="Weight decay for classifier training")  


    # Checkpointing and logging mixture precision, temperature for contrastive loss
    parser.add_argument("--checkpoint_freq", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--log_freq", type=int, default=20,
                        help="Log every N iterations")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for contrastive loss")
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--is_distributed", action="store_true",
                        help="Enable distributed training")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use Automatic Mixed Precision")
    
    args = parser.parse_args()
    
    # Calculate effective batch size
    args.global_batch_size = args.batch_size * args.nnodes * args.nproc_per_node
    
    return args


# =============================================================================
# DISTRIBUTED INITIALIZATION
# =============================================================================

def init_distributed_mode(args):
    """Initialize distributed training for multiple nodes."""
    
    # Check if running under torchrun
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print(f"Running in single GPU mode on {socket.gethostname()}")
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        args.distributed = False
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return args
    
    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # FIX: Properly set node_rank from environment
    args.node_rank = int(os.environ.get("GROUP_RANK", args.node_rank))
    
    # Set device
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device(f"cuda:{args.local_rank}")
    
    # Initialize process group with timeout
    timeout = timedelta(minutes=30)
    
    # FIX: Better error handling for multi-node initialization
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            world_size=args.world_size,
            rank=args.rank,
            timeout=timeout
        )
    except Exception as e:
        logger.info(30*"=")
        logger.error(f"Failed to initialize process group: {e}")
        logger.error(f"Master: {args.master_addr}:{args.master_port}")
        logger.error(f"Rank: {args.rank}/{args.world_size}")
        sys.exit(1)
    
    # Synchronize all processes
    dist.barrier()
    
    # Verify all processes are connected
    if args.rank == 0:
        logger.info("\n" + "="*80)
        logger.info("MULTI-NODE DISTRIBUTED TRAINING INITIALIZED")
        logger.info("="*80)
        logger.info(f"Total Nodes: {args.nnodes}")
        logger.info(f"GPUs per Node: {args.nproc_per_node}")
        logger.info(f"Total Processes: {args.world_size}")
        logger.info(f"Global Batch Size: {args.global_batch_size}")
        logger.info(f"Master Address: {args.master_addr}:{args.master_port}")
        logger.info(f"Backend: nccl")
        logger.info("="*80 + "\n")
    
    # FIX: Proper node identification
    logger.info(f"Node {args.node_rank} | Host: {socket.gethostname()} | "
                f"Rank: {args.rank}/{args.world_size} | GPU: {args.local_rank} | "
                f"PID: {os.getpid()}")
    
    # FIX: Add tensor synchronization test to verify multi-node communication
    if args.rank == 0:
        test_tensor = torch.ones(1).to(args.device)
    else:
        test_tensor = torch.zeros(1).to(args.device)
    
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    
    if args.rank == 0:
        expected_sum = args.world_size
        actual_sum = test_tensor.item()
        if abs(actual_sum - expected_sum) < 0.001:
            logger.info(f"Multi-node communication test passed (sum={actual_sum})")
            logger.info(30*"=" + "\n")
        else:
            logger.warning(f"Communication test: expected {expected_sum}, got {actual_sum}")
    
    dist.barrier()
    
    return args

# =============================================================================
# UTILITY FUNCTIONS FOR MULTI-NODE
# =============================================================================
def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_available() and dist.is_initialized():
        # FIX: Add barrier before cleanup
        dist.barrier()
        dist.destroy_process_group()
        logger.info(f"🧹 Rank {os.environ.get('RANK', 0)}: Distributed cleanup complete")
        logger.info(30*"=" + "\n")
      
def is_main_process():
    """Check if current process is the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes (average)."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_tensors(tensor):
    """Gather tensors from all processes."""
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list

def synchronize():
    """Synchronize all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()