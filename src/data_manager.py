import os
import math
import shutil
import time
import pathlib
import csv
import io
from pathlib import Path
from importlib.resources import files
import torch
from torch.utils.data import Dataset, Sampler, Subset
from torch.utils.data import random_split
from PIL import Image
from glob import glob

from src.data_augmentation import ThermalAugmentation
from src.monitoring import ResourceMonitoringThread
from src_utils.logging import get_logger

logger = get_logger(__name__)

# =========================================================
# Distributed Balanced Sampler
# =========================================================

import math
import torch
from torch.utils.data import Sampler


class DistributedBalancedSampler(Sampler):

    def __init__(
        self,
        labels,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
        drop_last=False,
    ):
        self.labels = labels
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        labels_tensor = torch.tensor(labels)

        # -------------------------
        # gather class indices
        # -------------------------
        self.classes = sorted(labels_tensor.unique().tolist())

        self.class_indices = {
            c: torch.where(labels_tensor == c)[0]
            for c in self.classes
        }

        # smallest class count
        self.num_per_class = min(
            len(v) for v in self.class_indices.values()
        )

        self.total_size = (
            len(self.classes) * self.num_per_class
        )

        if self.drop_last:
            self.num_samples = (
                self.total_size // self.num_replicas
            )
        else:
            self.num_samples = int(
                math.ceil(
                    self.total_size / self.num_replicas
                )
            )

        logger.info(
            f"BalancedSampler | "
            f"classes={self.classes} "
            f"per_class={self.num_per_class} "
            f"total={self.total_size}"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        balanced_indices = []

        # -------------------------
        # sample same amount
        # from each class
        # -------------------------
        for c in self.classes:

            idx = self.class_indices[c]

            perm = torch.randperm(
                len(idx),
                generator=g
            )

            balanced_indices.append(
                idx[perm][:self.num_per_class]
            )

        indices = torch.cat(
            balanced_indices,
            dim=0
        )

        # global shuffle
        if self.shuffle:
            indices = indices[
                torch.randperm(
                    len(indices),
                    generator=g
                )
            ]

        # DDP shard
        indices = indices.tolist()
        indices = indices[
            self.rank::self.num_replicas
        ]

        return iter(indices)


# =========================================================
# Contrastive Collate
# =========================================================

def contrastive_collate_fn(batch):

    batch = [
        b for b in batch
        if b is not None
    ]

    images = []
    labels = []
    ids = []

    for (img1, img2), label, idx in batch:

        images.append((img1, img2))
        labels.append(label)
        ids.append(idx)

    # concatenate views -> [6, H, W]
    concatenated = [

        torch.cat([im1, im2], dim=0)

        for im1, im2 in images
    ]

    return {

        "images": torch.stack(concatenated),

        "labels": torch.tensor(
            labels,
            dtype=torch.long
        ),

        "image_ids": torch.tensor(
            ids,
            dtype=torch.long
        )
    }
def classification_collate_fn(batch):

    batch = [
        b for b in batch
        if b is not None
    ]

    images = []
    labels = []

    for img, label in batch:

        images.append(img)
        labels.append(label)
        

    return {

        "images": torch.stack(images),

        "labels": torch.tensor(
            labels,
            dtype=torch.long
        )
    }


# =========================================================
# Dataset
# =========================================================
class SpillDataset(Dataset):

    def __init__(self, root_dir, isContrastive=True):

        self.transform = ThermalAugmentation(image_size=224)
        self.isContrastive = isContrastive

        self.image_paths = []
        self.labels = []

        label_map = {
            "clean": 0,
            "warm": 1,
            "cold": 2,
        }

        for class_name, label in label_map.items():

            class_dir = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_dir):
                logger.warning(f"Missing folder: {class_dir}")
                continue

            files = (
                glob(os.path.join(class_dir, "**", "*.jpg"), recursive=True) +
                glob(os.path.join(class_dir, "**", "*.png"), recursive=True) +
                glob(os.path.join(class_dir, "**", "*.jpeg"), recursive=True)
            )

            self.image_paths.extend(files)
            self.labels.extend([label] * len(files))

            logger.info(
                f"{class_name}: {len(files)} images"
            )

          
        logger.info(
            f"Dataset loaded: {len(self.image_paths)} images"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        label = self.labels[idx]

        if self.isContrastive:
            img1 = self.transform(img)
            img2 = self.transform(img)
            return (img1, img2), label, idx

        img = self.transform(img)
        return img, label
# =========================================================
# CSV Logger
# =========================================================

class CSVLogger:

    def __init__(self, fname, header):

        self.fname = fname

        self.buffer = io.StringIO()

        self.writer = csv.writer(
            self.buffer,
            quoting=csv.QUOTE_NONNUMERIC
        )

        self.writer.writerow(header)

        self.initialized = False

    def writerow(self, row):

        self.writer.writerow(row)

    def flush(self):

        mode = "a+" if self.initialized else "w"

        with open(
            self.fname,
            mode,
            newline=""
        ) as f:

            f.write(self.buffer.getvalue())

        self.buffer = io.StringIO()

        self.writer = csv.writer(
            self.buffer,
            quoting=csv.QUOTE_NONNUMERIC
        )

        self.initialized = True


# =========================================================
# Monitored Dataset
# =========================================================

class MonitoredDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset,
        log_filename,
        log_interval,
        monitor_interval
    ):

        self.dataset = dataset

        self.log_filename = str(log_filename)

        self.log_interval = log_interval
        self.monitor_interval = monitor_interval

        self._csv_log = None
        self._monitoring_thread = None
        self._last_log_time = None

        self.labels = self._get_labels(dataset)

    def _get_labels(self, ds):

        if hasattr(ds, "labels"):

            if ds.labels is not None:
                return ds.labels

        if hasattr(ds, "dataset"):

            return self._get_labels(ds.dataset)

        return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        self.maybe_start_resource_monitoring()

        return self.dataset[index]

    def _elapsed_log_time(self):

        if self._last_log_time is None:
            return float("inf")

        return (
            time.perf_counter() -
            self._last_log_time
        )

    def _update_log_time(self):

        self._last_log_time = time.perf_counter()

    def maybe_start_resource_monitoring(self):

        if self._monitoring_thread is not None:
            return

        def callback_fn(resource_sample):

            worker_info = (
                torch.utils.data.get_worker_info()
            )

            worker_id = (
                worker_info.id
                if worker_info is not None
                else 0
            )

            if self._csv_log is None:

                header = [
                    f.name
                    for f in resource_sample.fields()
                ]

                log_filename = (
                    self.log_filename.replace(
                        "%w",
                        str(worker_id)
                    )
                )

                self._csv_log = CSVLogger(
                    log_filename,
                    header
                )

            row_values = resource_sample.as_tuple()

            self._csv_log.writerow(row_values)

            if (
                self._elapsed_log_time()
                > self.log_interval
            ):

                self._csv_log.flush()

                self._update_log_time()

        self._monitoring_thread = (
            ResourceMonitoringThread(
                None,
                self.monitor_interval,
                stats_callback_fn=callback_fn
            )
        )

        self._monitoring_thread.start()

    def stop_resource_monitoring(self):

        if self._monitoring_thread:

            self._monitoring_thread.stop()

    def __del__(self):

        self.stop_resource_monitoring()


# =========================================================
# Worker Info
# =========================================================

def get_worker_info():

    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:

        num_workers = 1
        worker_id = 0

    else:

        num_workers = worker_info.num_workers
        worker_id = worker_info.id

    return num_workers, worker_id


# =========================================================
# Init Data
# =========================================================

def init_data(
    root_dir,
    rank=0,
    compute_loss_mode = "full",
    isContrastive:bool = True,
    num_workers=8,
    batch_size=64,
    drop_last=True,
    pin_mem=True,
    persistent_workers=True,
    world_size=1
):

    # =====================================================
    # Dataset
    # =====================================================

    dataset = SpillDataset(
        root_dir=root_dir,
        isContrastive=isContrastive
    )

    dataset_size = len(dataset)

    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(

        dataset,

        [train_size, val_size],

        generator=torch.Generator().manual_seed(42)
    )

    # =====================================================
    # Labels for subsets
    # =====================================================

    train_labels = [
        dataset.labels[i]
        for i in train_dataset.indices
    ]

    val_labels = [
        dataset.labels[i]
        for i in val_dataset.indices
    ]

    # =====================================================
    # Monitoring
    # =====================================================
    folder_name = "contrastive" if isContrastive else "classification"

    log_dir = (
        files("lr_monitoring_csv.ressource")
        / compute_loss_mode
        / folder_name
    )


    if log_dir.exists():
        shutil.rmtree(log_dir)

    log_dir.mkdir( parents=True,exist_ok=True)    

    train_dataset = MonitoredDataset(
        dataset=train_dataset,

        log_filename=str(
                log_dir /
                f"train_resource_{rank}_%w.csv"
            ),

        log_interval=10.0,

            monitor_interval=5.0,
        )

    val_dataset = MonitoredDataset(

            dataset=val_dataset,

            log_filename=str(
                log_dir /
                f"val_resource_{rank}_%w.csv"
            ),

            log_interval=10.0,

            monitor_interval=5.0,
        )

    # =====================================================
    # Samplers
    # =====================================================

    train_sampler = DistributedBalancedSampler(

        labels=train_labels,

        num_replicas=world_size,

        rank=rank,
        drop_last=drop_last,
        shuffle=True,
    )

    val_sampler = DistributedBalancedSampler(

        labels=val_labels,

        num_replicas=world_size,

        rank=rank,
        drop_last= drop_last if drop_last else False,

        shuffle=False,
    )

    # =====================================================
    # Loaders
    # =====================================================

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        sampler=train_sampler,

        batch_size=batch_size,

        collate_fn=contrastive_collate_fn if isContrastive else classification_collate_fn
        ,

        drop_last=drop_last,

        pin_memory=pin_mem,

        num_workers=num_workers,

        persistent_workers=(
            (num_workers > 0)
            and persistent_workers
        ),
    )

    val_loader = torch.utils.data.DataLoader(

        val_dataset,

        sampler=val_sampler,

        batch_size=batch_size,

        collate_fn=contrastive_collate_fn if isContrastive else classification_collate_fn
        ,

        drop_last=False,

        pin_memory=pin_mem,

        num_workers=num_workers,

        persistent_workers=(
            (num_workers > 0)
            and persistent_workers
        ),
    )

    logger.info(
        "Dataset and loader created successfully."
    )

    return (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler
    )
    
    
# =========================================================
# Test Init Data
# =========================================================

from typing import Tuple
import torch
from torch.utils.data import random_split

def test_init_data(
    test_data_dir: str,
    rank: int = 0,
    isContrastive: bool = True,
    num_workers: int = 4,
    batch_size: int = 64,
    drop_last: bool = False,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    world_size: int = 1,
    shuffle: bool = True,
) -> Tuple[torch.utils.data.DataLoader, "DistributedBalancedSampler"]:

    # =====================================================
    # Dataset
    # =====================================================
    test_dataset = SpillDataset(
        root_dir=test_data_dir,
        isContrastive=isContrastive
    )

    dataset_size = len(test_dataset)
   

    train_size = int(0.85 * dataset_size)
    test_size = dataset_size - train_size

    # =====================================================
    # Split dataset
    # =====================================================
    train_subset, test_subset = random_split(
        test_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # IMPORTANT: labels must come from subset indices
    test_labels = [
        test_dataset.labels[i]
        for i in test_subset.indices
    ]

    # =====================================================
    # Sampler
    # =====================================================
    test_sampler = DistributedBalancedSampler(
        labels=test_labels,
        num_replicas=world_size,
        rank=rank,
        drop_last=drop_last,
        shuffle=shuffle,
    )

    # =====================================================
    # Collate
    # =====================================================
    collate_fn = (
        contrastive_collate_fn
        if isContrastive
        else classification_collate_fn
    )

    # =====================================================
    # DataLoader
    # =====================================================
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        sampler=test_sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0 and persistent_workers),
    )

    logger.info(
        f"Test loader created | "
        f"samples={len(test_subset)} | "
        f"batch_size={batch_size} | "
        f"workers={num_workers}"
    )

    return test_loader