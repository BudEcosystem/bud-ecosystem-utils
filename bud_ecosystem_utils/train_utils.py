import os
import time
import logging
from accelerate.state import PartialState

from bud_ecosystem_utils.blob import BlobService


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    An adapter to assist with logging in multiprocess.

    `log` takes in an additional `main_process_only` kwarg, which dictates whether it should be called on all processes
    or only the main executed one. Default is `main_process_only=True`.

    Does not require an `Accelerator` object to be created first.
    """
    LAST_LOGGED_AT = None
    BLOB_SERVICE = BlobService()

    @staticmethod
    def _should_log(main_process_only):
        "Check if log should be performed"
        state = PartialState()
        return not main_process_only or (main_process_only and state.is_main_process)

    def log(self, level, msg, *args, **kwargs):
        """
        Delegates logger call after checking if we should log.

        Accepts a new kwarg of `main_process_only`, which will dictate whether it will be logged across all processes
        or only the main executed one. Default is `True` if not passed

        Also accepts "in_order", which if `True` makes the processes log one by one, in order. This is much easier to
        read, but comes at the cost of sometimes needing to wait for the other processes. Default is `False` to not
        break with the previous behavior.

        `in_order` is ignored if `main_process_only` is passed.
        """
        if PartialState._shared_state == {}:
            raise RuntimeError(
                "You must initialize the accelerate state by calling either `PartialState()` or `Accelerator()` before using the logging utility."
            )
        main_process_only = kwargs.pop("main_process_only", True)
        in_order = kwargs.pop("in_order", False)
        blob_key = kwargs.pop("blob_key", self.extra.get("blob_key", None))
        is_last_msg = kwargs.pop("end", False)

        if self.isEnabledFor(level):
            publish_log = is_last_msg or self.LAST_LOGGED_AT is None or time.time() - self.LAST_LOGGED_AT >= int(os.environ.get("LOG_PUBLISH_INTERVAL", 30))
            if self._should_log(main_process_only):
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)
                if publish_log:
                    self.BLOB_SERVICE.upload_file(blob_key, filepath=self.logger.root.handlers[0].baseFilename)
                    self.LAST_LOGGED_AT = time.time()
            elif in_order:
                state = PartialState()
                for i in range(state.num_processes):
                    if i == state.process_index:
                        msg, kwargs = self.process(msg, kwargs)
                        self.logger.log(level, msg, *args, **kwargs)
                        if publish_log:
                            self.BLOB_SERVICE.upload_file(blob_key, filepath=self.logger.root.handlers[0].baseFilename)
                            self.LAST_LOGGED_AT = time.time()
                    state.wait_for_everyone()
