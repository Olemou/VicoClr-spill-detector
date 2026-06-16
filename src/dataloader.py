
import bisect
import csv
import io
import time

import numpy as np
import torch
from torch.utils.data import _utils
from torch.utils.data.dataloader import ExceptionWrapper, _DatasetKind, _MultiProcessingDataLoaderIter

from src.monitoring import ResourceMonitoringThread
from torch.utils.data import Dataset


class CSVLogger(object):
    """An append-to CSV abstraction. File I/O requires a flush."""
    def __init__(self, fname, header):
        self.fname = fname
        self.buffer = io.StringIO()
        self.writer = csv.writer(self.buffer, quoting=csv.QUOTE_NONNUMERIC)
        self.writer.writerow(header)
        self.initialized = False

    def writerow(self, row) -> None:
        self.writer.writerow(row)

    def flush(self) -> None:
        mode = "a+" if self.initialized else "w"
        with open(self.fname, mode, newline="") as f:
            f.write(self.buffer.getvalue())
        self.buffer = io.StringIO()
        self.writer = csv.writer(self.buffer, quoting=csv.QUOTE_NONNUMERIC)
        self.initialized = True


class MonitoredDataset(Dataset):
    """Resource monitoring wrapper that properly handles Subset + labels."""

    def __init__(
        self, 
        dataset: Dataset, 
        log_filename: str, 
        log_interval: float = 10.0, 
        monitor_interval: float = 5.0
    ):
        self.dataset = dataset
        self.log_filename = str(log_filename)
        self.log_interval = log_interval
        self.monitor_interval = monitor_interval
        
        self._csv_log = None
        self._monitoring_thread = None
        self._last_log_time = None

        # Critical: Recursively extract labels from Subset
        self.labels = self._get_labels(dataset)

    def _get_labels(self, ds):
        """Recursively find .labels even inside Subset / wrappers"""
        if hasattr(ds, 'labels') and ds.labels is not None:
            return ds.labels
        if hasattr(ds, 'label') and ds.label is not None:
            return ds.label
        if hasattr(ds, 'dataset'):  # Subset, MonitoredDataset, etc.
            return self._get_labels(ds.dataset)
        return None

    def __getitem__(self, index):
        self.maybe_start_resource_monitoring()
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError
        return getattr(self.dataset, name)

    def _elapsed_log_time(self):
        if self._last_log_time is None:
            return float("inf")
        return time.perf_counter() - self._last_log_time

    def _update_log_time(self):
        self._last_log_time = time.perf_counter()

    def maybe_start_resource_monitoring(self):
        if self._monitoring_thread is not None:
            return

        def callback_fn(resource_sample):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0

            if self._csv_log is None:
                header = [f.name for f in resource_sample.fields()]
                log_filename = self.log_filename.replace("%w", str(worker_id))
                self._csv_log = CSVLogger(log_filename, header)

            row_values = resource_sample.as_tuple()
            self._csv_log.writerow(row_values)

            if self._elapsed_log_time() > self.log_interval:
                self._csv_log.flush()
                self._update_log_time()

        self._monitoring_thread = ResourceMonitoringThread(
            None, self.monitor_interval, stats_callback_fn=callback_fn
        )
        self._monitoring_thread.start()

    def stop_resource_monitoring(self):
        if self._monitoring_thread:
            self._monitoring_thread.stop()
            self._monitoring_thread = None

    def __del__(self):
        self.stop_resource_monitoring()


def get_worker_info():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        num_workers = 1
        worker_id = 0
    else:
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
    return num_workers, worker_id