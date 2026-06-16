from html import parser
import os
import time
from importlib.resources import files
from pathlib import Path
from src.utils import DataIterator
import os
import shutil
import gc
import warnings
import numpy as np
import torch
import torch.nn as nn
from src_utils.logging import get_logger
from src.data_manager import init_data
from src.loss import  supervised_contrastive_loss,co_cluster_opinion_batch
from src_utils.lr_scheduler import get_optimizer_params
from src_utils.logging import gpu_timer, CSVLogger
from src_utils.monitoring import AverageMeter
from src_utils.lr_scheduler import cosine_schedule,log_training_status
from src.model import VisionTransformer, default_load_pretrained_weights, load_checkpoint
from src.ddp import init_distributed_mode, is_main_process,setup_environment, seed_everything,parse_arguments
logger = get_logger("DDP Training",force=True)
import warnings
warnings.filterwarnings("error")  
warnings.filterwarnings("ignore", category=DeprecationWarning)


#
def train_contrastive(args):

# -------------------------------
    # Setup =================================================================================================)
    
    #Verify distributed setup before proceeding
    

    #=====================================================================================================
    root_dir = args.root_dir
    #===================Rank, batch_size, num_workers,world_size====================================================================
    rank = args.rank
    batch_size = args.global_batch_size // args.world_size
    world_size = args.world_size
    num_workers = args.num_workers
    
   #=================nbre of epochs, warmup, save_every_freq====================================================================
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    scaler = torch.amp.GradScaler()
    sync_gc = True,
    GARBAGE_COLLECT_ITR_FREQ=20

    #\================================================================================================
    loss_meter = AverageMeter()
    iter_time_meter = AverageMeter()
    gpu_time_meter = AverageMeter()
    data_elapsed_time_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    diff_weight_mean_meter = AverageMeter()
    neg_weight_mean_meter = AverageMeter()
    
    diff_weight_mean_meter_eval = AverageMeter()
    neg_weight_mean_meter_eval = AverageMeter()

    #===========================
    loss_reg_std_mult = args.loss_reg_std_mult
    loss_reg_min_epoch = args.loss_reg_min_epoch
    loss_reg_num_tracking_steps = args.loss_reg_num_tracking_steps
    
    #================================================================================================  
    #Data Loading and Dataloader setup   
    #================================================================================================  
    train_loader, val_loader ,train_sampler, val_sampler= init_data(
        root_dir=root_dir, rank=rank, 
        num_workers=num_workers,
        batch_size=batch_size, drop_last=True, 
        pin_mem=True, persistent_workers=True, 
        world_size=world_size,
        isContrastive=True,
        compute_loss_mode=args.compute_loss_mode,
    )
    
    steps_per_epoch = len(train_loader)
    log_freq = args.log_freq
    CHECKPOINT_FREQ  = args.checkpoint_freq
    #=============================================================================================================================
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    #================================================================================
    check_point_folder = files("checkpoint")
    
    if is_main_process():
        checkpoint_path = (
            check_point_folder
            / "contrastive"
            / args.compute_loss_mode
            / f"checkpoint_r{rank}.pth"
        )
        
    #===============================loading Model =================================================================
    online_model = VisionTransformer(model_size=args.model_size,patch_dropout_prob=0.2)
    optimizer = get_optimizer_params(online_model, batch_size)
    
    if args.continuing_training:
        start_epoch = load_checkpoint(checkpoint_path, online_model, optimizer=optimizer)
        online_model.to(args.device)
        logger.info("continuous training")
    else:
        start_epoch = 0
        online_model = default_load_pretrained_weights(online_model, device=args.device)
        logger.info("begining training mode")
        

    

    #================================================================================
     # -- monitoring paths
      # -- monitoring paths
      
    def setup_monitoring():
         if not is_main_process():
             return
         
         logger.info(f"Setting up monitoring directories for rank {rank}...")
         
         train_dir = (
             files("lr_monitoring_csv")
             /"train"
             / args.compute_loss_mode
             / "contrastive"
         )
         
         if train_dir.exists():
             shutil.rmtree(train_dir)

         train_dir.mkdir(
                 parents=True,
                 exist_ok=True
             )
         train_log_file = os.path.join(train_dir, f"log_r{rank}.csv")
         
         csv_logger_train = CSVLogger( train_log_file,
         ("%d", "epoch"),
         ("%d", "itr"),
         ("%.5f", "loss"),
         ("%d", "gpu-time(ms)"),
         ("%d", "dataload-time(ms)"),
         ("%d", "diff_weight_mean"),
         ("%d", "neg_weight_mean"), )
         
         return csv_logger_train
     
           
#================================Iterator====================================================
    csv_logger_train =  setup_monitoring()
    data_iter_train = DataIterator(train_loader, sampler=train_sampler)
    data_iter_val = DataIterator(val_loader,val_sampler)
    mixed_precision = True


    training_losses = []
    step_count = 0
    # ----- Garbage collection before batch -----
    if sync_gc:
        gc.disable()
        gc.collect()
    
    for epoch in range(start_epoch, num_epochs):
        online_model.train()
        data_iter_train.set_epoch(epoch)
        data_iter_val.set_epoch(epoch)
        
        # ================================================================
        logger.info("Training Start For Epoch %d" % (epoch + 1))
        #=================================================================================================
        loss_meter.reset()
        iter_time_meter.reset()
        gpu_time_meter.reset()
        val_loss_meter.reset()
        data_elapsed_time_meter.reset()
        diff_weight_mean_meter.reset()
        neg_weight_mean_meter.reset()
        diff_weight_mean_meter_eval.reset()
        neg_weight_mean_meter_eval.reset()
        
        
        # ============================
        cosine_schedule(epoch, optimizer, warmup_epochs=warmup_epochs, max_epochs=num_epochs)
        
        # ============================ save checkpoints =========================================
        
        def save_checkpoint(epoch: int):
            """
            Save checkpoint. Works with single GPU and multi-GPU (DDP).
            Saves model (encoder) + optimizer + training state.
            """
            if not is_main_process():
                return
            path  = (check_point_folder / "contrastive"
            / args.compute_loss_mode   
            )
            
            if path.exists():
                shutil.rmtree(path) 
                
            path.mkdir(parents=True, exist_ok=True)


            # Handle DDP vs single GPU
            if hasattr(online_model, 'module'):          # DDP wrapped model
                model_state = online_model.module.state_dict()
            else:
                model_state = online_model.state_dict()

            save_dict = {
                "epoch": epoch,
                "encoder": model_state,                    # ← Main weights (recommended key)
                "optimizer": optimizer.state_dict(),
                "loss": loss_meter.avg if 'loss_meter' in globals() else None,
                "batch_size": batch_size,
                "world_size": world_size,
                "compute_loss_mode": args.compute_loss_mode
            }

            try:
                # Ensure path ends with .pth
                
                torch.save(save_dict, path / f"checkpoint_r{rank}.pth")
                logger.info(f"Checkpoint saved successfully at epoch {epoch}: {path}")
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                
        # ============================ training loop =========================================
        
        for itr in range(steps_per_epoch):
    
            itr_start_time = time.time()
            #===========================
            # Fetch batch safely
            batch = data_iter_train.next(epoch)

            if isinstance(batch, (list, tuple,dict)):

                inputs = batch["images"]
                labels = batch["labels"]
               
                idxs = batch["image_ids"]
                xi = inputs[:, :3]
                xj = inputs[:, 3:]
                inputs = torch.cat([xi, xj], dim=0).to(args.device, non_blocking=True)
                targets = torch.cat([labels, labels], dim=0).to(args.device, non_blocking=True)
                img_ids = torch.cat([idxs, idxs], dim=0).to(args.device, non_blocking=True)
            else:
                inputs = batch.to(args.device)
                targets = None
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            
            
           #===========================================================

           # ----- Periodic garbage collection -----
            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()
            # ---------- Forward and backward ----------
            def train_forward_step():
                
                with torch.amp.autocast(dtype=torch.bfloat16, enabled=mixed_precision,device_type=args.device.type):
                  outputs = online_model(inputs)
                  outputs = torch.nn.functional.normalize(outputs, dim=-1)
                 
                  uncertainty = co_cluster_opinion_batch( outputs, targets )
                  
                  loss,stats_train = supervised_contrastive_loss(
                    z=outputs,
                    u = uncertainty,
                    labels=targets,
                    img_ids=img_ids,
                    epoch=epoch,
                    temperature=args.temperature,
                    same_img_weight=1.0,
                    eps=1e-8,
                    T=num_epochs,
                    mode=args.compute_loss_mode,
                    return_stats=True,
                )
                diff_weight_mean_meter.update(stats_train['diff_weight_mean'])
                neg_weight_mean_meter.update(stats_train['neg_weight_mean'])
                # Step 2. Backward & step
                run_step = True
                if args.compute_loss_mode == "":
                    
                    if loss_reg_std_mult is not None and len(training_losses) > 0 :
                        meanval = np.mean(training_losses)
                        stdval = np.std(training_losses)
                        max_bound = meanval + loss_reg_std_mult * stdval
                        if (loss > max_bound and epoch > loss_reg_min_epoch and len(training_losses)> int(0.5 * loss_reg_num_tracking_steps)):
                            run_step = False
                            loss.backward()
                            logger.info(
                                    f"Loss {loss} is above bound {meanval} + {loss_reg_std_mult} * {stdval}. Skipping step."
                                )
                        if run_step:
                            if mixed_precision:
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                            else:
                                loss.backward()
                            if mixed_precision:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                        if (itr % log_freq == 0 and is_main_process()):
                            log_training_status(optimizer, online_model, epoch, step=None)
                        optimizer.zero_grad()
                else:
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    if mixed_precision:  
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if (itr % log_freq == 0 and is_main_process()):
                        log_training_status(optimizer, online_model, epoch, step=None)
                    optimizer.zero_grad()
                   

                return  (
                  loss.detach().item(),
                    run_step,
                )
            (loss,run_step), gpu_time_ms = gpu_timer(train_forward_step)
            loss_meter.update(loss)
            gpu_time_meter.update(gpu_time_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)
            #=================================================================================
            if loss_reg_std_mult is not None and args.compute_loss_mode == "":
                if run_step:
                    training_losses.append(loss)
                    if len(training_losses) > loss_reg_num_tracking_steps:
                        training_losses = training_losses[1:]
                        step_count = 0
                else:
                    step_count += 1
                    if step_count > steps_per_epoch // 2:
                        raise RuntimeError(
                            "Loss is above bound for too many tries. Exiting."
                        )
            #===================================================================================== # -- Logging
            def log_stats():
                    if not is_main_process():
                      return
                     
                    if itr == steps_per_epoch - 1:
                        csv_logger_train.log(
                        epoch + 1,
                        itr,
                        loss_meter.avg,
                        gpu_time_meter.avg,
                        data_elapsed_time_meter.avg,
                        diff_weight_mean_meter.avg,
                        neg_weight_mean_meter.avg
                    )
                    if (
                        (itr % log_freq == 0)
                        or np.isnan(loss)
                        or np.isinf(loss)
                        ):
                        logger.info(
                            "[iter: %.1f ms] "
                            "[%d, %5d] loss: %.3f "
                            "[gpu: %.1f ms] "
                            "[data: %.1f ms] "
                            "diff_weight_mean: %.3f "
                            "neg_weight_mean: %.3f"
                            % (
                                iter_time_meter.avg,
                                epoch + 1,
                                itr,
                                loss_meter.avg,
                                gpu_time_meter.avg,
                                data_elapsed_time_meter.avg,
                                diff_weight_mean_meter.avg,
                                neg_weight_mean_meter.avg
                            ))
            log_stats() 
        
        #=====================================================================================================
           # -- Save Checkpoint 
        if is_main_process():
            logger.info("avg loss Training %.3f" % loss_meter.avg)
        if (epoch + 1) % 5 == 0  or epoch == (args.num_epochs - 1):
            save_checkpoint(epoch + 1)
            
        if is_main_process():
            logger.info(60*"=" + "\n")
        
        if sync_gc:
            gc.enable()
        logger.info(60*"=" + "\n")                                       
            
            
  
