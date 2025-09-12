import logging
import os
from pythonjsonlogger.json import JsonFormatter


class CustomJsonFormatter(JsonFormatter):
    """Formats log lines in JSON."""

    def process_log_record(self, log_record: dict) -> dict:
        """Modifies fields in the log_record to match Cloud Logging's expectations."""
        log_record["severity"] = log_record.get("levelname", "INFO")
        created = float(log_record.get("created", 0.0))
        log_record["timestampSeconds"] = int(created)
        log_record["timestampNanos"] = int((created % 1) * 1_000_000_000)
        return log_record


def configure_logger(level: str | None = None, json_format: bool = True):
    """Configures python root logger safely.

    This function avoids attaching duplicate StreamHandlers if called multiple times
    (common in tests) and sets the formatter according to json_format.
    """
    root_logger = logging.getLogger()
    if level is not None:
        root_logger.setLevel(getattr(logging, level.upper()))
    else:
        root_logger.setLevel(os.getenv("LOG_LEVEL", "WARNING"))

    # Find existing stream handler if present
    existing_stream_handler = None
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler):
            existing_stream_handler = h
            break

    if json_format:
        formatter = CustomJsonFormatter(
            "%(name)s|%(levelname)s|%(message)s|%(created)f|%(lineno)d|%(pathname)s",
            "%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(lineno)d|%(pathname)s|%(name)s|%(levelname)s:%(message)s"
        )

    if existing_stream_handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        existing_stream_handler.setFormatter(formatter)

    # Avoid propagating to ancestor handlers if already configured elsewhere
    root_logger.propagate = False
