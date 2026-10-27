# shared_logger.py
import logging
import os
from threading import Lock


class SharedLogger:
    """Simple shared logger that creates one log file in /tmp/ and can be used by multiple modules"""

    _loggers = {}
    _lock = Lock()

    def __init__(self, name="processing", logger_name="shared_logger"):
        self.name = name
        self.logger_name = logger_name
        self.log_file = f"/tmp/processing_{name}.log"

        with SharedLogger._lock:
            if logger_name not in SharedLogger._loggers:
                base_logger = self._create_logger()
                # Wrap logger in an adapter that injects logger_name
                SharedLogger._loggers[logger_name] = logging.LoggerAdapter(base_logger, {"logger_name": logger_name})

            self.logger = SharedLogger._loggers[logger_name]

    def _create_logger(self):
        """Create and configure logger"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # File handler
        handler = logging.FileHandler(self.log_file, mode='a')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(logger_name)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)
