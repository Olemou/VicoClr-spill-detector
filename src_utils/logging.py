
import torch 
import os
import sys
import logging
LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
  
def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)  

def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class CSVLogger(object):

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.types = []
        mode = kwargs.get("mode", "+a")
        self.delim = kwargs.get("delim", ",")
        # -- print headers
        with open(self.fname, mode) as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=self.delim, file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv):
        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = self.delim if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)