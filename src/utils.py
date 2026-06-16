import time
import torch            
from src_utils.logging import get_logger
logger = get_logger(__name__)


class DataIterator:
    def __init__(self, dataloader, sampler=None, max_retries=5, retry_sleep=5):
        self.dataloader = dataloader
        self.sampler = sampler
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.loader = iter(dataloader)

    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def reset(self):
        self.loader = iter(self.dataloader)

    def next(self, epoch):
        iter_retries = 0
        iter_successful = False
        while not iter_successful:
            try:
                sample = next(self.loader)
                iter_successful = True
                return sample

            except StopIteration:
                logger.info("Exhausted dataset. Restarting loader...")
                self.set_epoch(epoch)
                self.reset()

            except Exception as e:
                if iter_retries < self.max_retries:
                    iter_retries += 1
                    logger.warning(
                        f"Data loading error retry {iter_retries}/{self.max_retries}: {e}"
                    )
                    time.sleep(self.retry_sleep)
                else:
                    raise RuntimeError(
                        f"Failed loading data after {self.max_retries} retries"
                    ) from e
                    
                