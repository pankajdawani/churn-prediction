"""
Centralized logger for the churn prediction project.
Outputs structured logs to console and a rotating file.
"""
import logging
import logging.handlers
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "churn_prediction.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with console + rotating file handlers.

    Args:
        name: Logger name (use __name__ from the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Rotating file handler (5 MB max, keep 3 backups) ─────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent log messages from propagating to the root logger
    logger.propagate = False

    return logger
