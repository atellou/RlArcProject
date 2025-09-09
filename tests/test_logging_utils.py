import logging
from rlarcworld.utils.logging_utils import configure_logger


def test_configure_logger_no_duplicate_handlers():
    root = logging.getLogger()
    # Remove existing handlers to start clean
    for h in list(root.handlers):
        root.removeHandler(h)

    configure_logger(level="INFO", json_format=False)
    handlers_after_first = list(root.handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in handlers_after_first)

    # Call again; should reuse existing StreamHandler, not add a new one
    configure_logger(level="INFO", json_format=False)
    handlers_after_second = list(root.handlers)
    assert len(handlers_after_second) == len(handlers_after_first)


def test_configure_logger_switch_format():
    root = logging.getLogger()
    # Ensure there is one handler
    configure_logger(level="INFO", json_format=False)
    n_handlers_before = len(root.handlers)

    # Switch to JSON formatting; should not increase handler count
    configure_logger(level="INFO", json_format=True)
    assert len(root.handlers) == n_handlers_before
