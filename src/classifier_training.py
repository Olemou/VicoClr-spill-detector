import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src_utils.lr_scheduler import log_training_status
from src_utils.monitoring import AverageMeter
from src.utils import get_logger
from src.model import VisionTransformer, load_checkpoint
from pathlib import Path
from importlib.resources import files
import torch
import gc
import numpy as np
from src.utils import DataIterator
import shutil
import os
from src.vit_classifier import ViTClassifier
from src_utils.logging import CSVLogger, gpu_timer
from src.ddp import init_distributed_mode, is_main_process,setup_environment, seed_everything,parse_arguments
from src.data_manager import init_data
logger = get_logger("Classifier Training", force=True)


def train_classifier(args):
    # === DDP Setup ===
    
    

    #===================Rank, batch_size, num_workers,world_size====================================================================
    rank = args.rank
    batch_size = args.global_batch_size // args.world_size
    world_size = args.world_size
    num_workers = args.num_workers
    
   #=================nbre of epochs, warmup, save_every_freq====================================================================
    num_epochs = args.numbre_epoch_classifier
    scaler = torch.amp.GradScaler()
    sync_gc = True,
    GARBAGE_COLLECT_ITR_FREQ=10
    mixed_precision  = args.mixed_precision
    #=================training setup====================================================================
    
    #\================================================================================================
    classifier_train_loss_loss_meter = AverageMeter()
    classifier_val_loss_meter = AverageMeter()
    classifier_gpu_time_meter = AverageMeter()
  
    #===========================
    loss_reg_std_mult = args.loss_reg_std_mult
    loss_reg_min_epoch = args.loss_reg_min_epoch
    loss_reg_num_tracking_steps = args.loss_reg_num_tracking_steps
    
    #===================================================================

    check_point_folder_vit_clr = files("checkpoint")
    check_point_folder_vit_classifier = files("checkpoint")

    if is_main_process():
        checkpoint_path = (
            check_point_folder_vit_clr / "contrastive"
            / args.compute_loss_mode
            / f"checkpoint_r{rank}.pth"
        )
    checkpoint_path = Path(checkpoint_path)
    model_size = args.model_size
    vit_model = VisionTransformer(model_size=model_size)
    _ = load_checkpoint(checkpoint_path, vit_model)
    
    classifier_model = ViTClassifier(vit_model=vit_model, model_size=model_size, freeze_vit=True, dropout=0.2)
    classifier_model.to(args.device)
    
    logger.info(f"Initialized ViTClassifier with {model_size} backbone and loaded pretrained weights")
    
    
    #================================================================================
     # -- monitoring paths
    def setup_monitoring():
        if is_main_process():
          train_dir = (
            files("lr_monitoring_csv")
            /"train"
            / args.compute_loss_mode
            / "classification"
        )
        
        
        if train_dir.exists():
            shutil.rmtree(train_dir)

        train_dir.mkdir( parents=True, exist_ok=True )
        train_log_file = os.path.join(train_dir, f"log_r{rank}.csv")
        csv_logger_train_classifier = CSVLogger(
        train_log_file, ("%d", "epoch"), ("%d", "itr"), ("%.5f", "loss"), ("%d", "gpu-time(ms)"))
        return csv_logger_train_classifier
        
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_loader_classifier, val_loader_classifier , train_sampler_classifier, val_sampler_classifier= init_data(
            root_dir=args.root_dir, 
            rank=rank,
            num_workers=num_workers, 
            batch_size=batch_size, drop_last=True, 
            pin_mem=True, 
            persistent_workers=True, 
            world_size=world_size,
            isContrastive=False,
            compute_loss_mode=args.compute_loss_mode
        )
    data_iter_train_classifier = DataIterator(train_loader_classifier, sampler=train_sampler_classifier)
    data_iter_val_classifier = DataIterator(val_loader_classifier,val_sampler_classifier)
        
    steps_per_epoch = len(train_loader_classifier)
    log_freq = args.log_freq
    CHECKPOINT_FREQ  = args.checkpoint_freq
    csv_logger_train_classifier =setup_monitoring()
    #================================================================================
    optimizer_classifier = AdamW(classifier_model.parameters(), lr=args.lr_classifier, weight_decay=args.weight_decay_classifier)
    scheduler = CosineAnnealingLR(optimizer_classifier, T_max=num_epochs * len(train_loader_classifier), eta_min= args.lr_classifier * 0.01)
    
    
    #===========================================================================================================
    classifier_training_losses = []
    step_count = 0
    # ----- Garbage collection before batch -----
    if sync_gc:
        gc.disable()
        gc.collect()
    
    for epoch in range(0, num_epochs):
        classifier_model.train()
        data_iter_train_classifier.set_epoch(epoch)
        data_iter_val_classifier.set_epoch(epoch)
        
        # ================================================================
        logger.info("Training Start Classifier For Epoch %d" % (epoch + 1))
        #=================================================================================================
        classifier_train_loss_loss_meter.reset()
        classifier_val_loss_meter.reset()
        classifier_gpu_time_meter.reset()
        
        
        # ============================ save checkpoints =========================================
        
        def save_checkpoint(epoch: int):
            """
            Save checkpoint. Works with single GPU and multi-GPU (DDP).
            Saves model (encoder) + optimizer + training state.
            """
            if not is_main_process():
                return
            path  = (check_point_folder_vit_classifier/"classifier"
            / args.compute_loss_mode  
            )
            
            if path.exists():
                shutil.rmtree(path) 
                
            path.mkdir(parents=True, exist_ok=True)

            # Handle DDP vs single GPU
            if hasattr(classifier_model, 'module'):          # DDP wrapped model
                model_state = classifier_model.module.state_dict()
            else:
                model_state = classifier_model.state_dict()

            save_dict = {
                "epoch": epoch,
                "encoder": model_state,                    # ← Main weights (recommended key)
                "optimizer": optimizer_classifier.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "loss": classifier_train_loss_loss_meter.avg if 'classifier_train_loss_loss_meter' in globals() else None,
                "batch_size": batch_size,
                "world_size": world_size,
                "compute_loss_mode": args.compute_loss_mode
            }

            try:
                torch.save(save_dict, path / f"checkpoint_r{rank}.pth")
                logger.info(f"Checkpoint saved successfully at epoch {epoch}: {path}")
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                
        
        # ============================ training loop =========================================for itr in range(steps_per_epoch):
    
            #===========================
        for itr in range(steps_per_epoch):
            # Fetch batch safely
            batch = data_iter_train_classifier.next(epoch)

            if isinstance(batch, (list, tuple,dict)):

                inputs = batch["images"]
                labels = batch["labels"]
                
                inputs = inputs.to(args.device, non_blocking=True)
                targets = labels.to(args.device, non_blocking=True)
                targets = targets
            
            else:
                inputs = batch.to(args.device)
                targets = None
           #===========================================================
           # ----- Periodic garbage collection -----
            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()
             # ---------- Forward and backward ----------
            def train_forward_step():
                
                #with torch.amp.autocast(dtype=dtype, enabled=mixed_precision,device_type=args.device.type):
                outputs = classifier_model(inputs)
                
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, targets)
                
                
                # Step 2. Backward & step
                run_step = True
                if  args.compute_loss_mode == "full":
                    
                    if loss_reg_std_mult is not None and len(classifier_training_losses) > 0 :
                        meanval = np.mean(classifier_training_losses)
                        stdval = np.std(classifier_training_losses)
                        max_bound = meanval + loss_reg_std_mult * stdval
                        if (loss > max_bound and epoch > loss_reg_min_epoch and len(classifier_training_losses)> int(0.5 * loss_reg_num_tracking_steps)):
                            run_step = False
                            loss.backward()
                            logger.info(
                                    f"Loss {loss} is above bound {meanval} + {loss_reg_std_mult} * {stdval}. Skipping step."
                                )
                        if run_step:
                            if mixed_precision:
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer_classifier)
                            else:
                                loss.backward()
                            if mixed_precision:
                                scaler.step(optimizer_classifier)
                                scaler.update()
                                scheduler.step()
                            else:
                                optimizer_classifier.step()
                                scheduler.step()
                    
                        optimizer_classifier.zero_grad()
                else:
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer_classifier)
                    else:
                        loss.backward()
                        if mixed_precision:
                            scaler.step(optimizer_classifier)
                            scaler.update()
                            scheduler.step()
                        else:
                            optimizer_classifier.step()
                            scheduler.step()
                    
                    optimizer_classifier.zero_grad()

                return  (
                  loss.detach().item(),
                    run_step,
                )
            (loss,run_step), gpu_time_ms = gpu_timer(train_forward_step)
            
            classifier_train_loss_loss_meter.update(loss)
            classifier_gpu_time_meter.update(gpu_time_ms)
    
            #=================================================================================
            if loss_reg_std_mult is not None and args.compute_loss_mode == "":
                if run_step:
                    classifier_training_losses.append(loss)
                    if len(classifier_training_losses) > loss_reg_num_tracking_steps:
                        classifier_training_losses = classifier_training_losses[1:]
                        step_count = 0
                else:
                    step_count += 1
                    if step_count > steps_per_epoch // 2:
                        raise RuntimeError(
                            "Loss is above bound for too many tries. Exiting."
                        )
            #===================================================================================== # -- Logging
            
            #===================================================================================== # -- Logging
            def log_classifier_stats():
                    if not is_main_process():
   
                      return
                   
                    if itr == steps_per_epoch - 1:
                        csv_logger_train_classifier.log(
                        epoch + 1,
                        itr,
                        classifier_train_loss_loss_meter.avg,
                        classifier_gpu_time_meter.avg,
                        
                    )
                    if (
                        (itr % log_freq == 0)
                        or np.isnan(loss)
                        or np.isinf(loss)
                        ):
                        logger.info(
                              f"[{epoch + 1}, {itr:5d}] "
                              f"loss: {classifier_train_loss_loss_meter.avg:.3f} "
                              f"[gpu: {classifier_gpu_time_meter.avg:.1f} ms]"
                          )
            log_classifier_stats() 
        
        #=====================================================================================================
           # -- Save Checkpoint 
        if is_main_process():
            logger.info("average loss classifier training %.3f" % classifier_train_loss_loss_meter.avg)
        if (epoch + 1) % CHECKPOINT_FREQ == 0 or epoch == (args.num_epochs - 1):
            save_checkpoint(epoch + 1)
            
        if is_main_process():
            logger.info(60*"=" + "\n")
        if sync_gc:
            gc.enable()                                                                                                                                     
            
