"""Shared logging helpers."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import sys

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a runtime dependency
    tqdm = None


_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


class _NullProgressBar:
    def update(self, _n: int = 1) -> None:
        return

    def set_description(self, _desc: str, refresh: bool = True) -> None:
        return

    def set_postfix_str(self, _s: str, refresh: bool = True) -> None:
        return

    def close(self) -> None:
        return


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if tqdm is not None:
                tqdm.write(message, file=self.stream)
            else:
                self.stream.write(message + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(level: str = "INFO", log_path: str | Path | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    console_handler = TqdmLoggingHandler(stream=sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(console_handler)

    logging.captureWarnings(True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(numeric_level)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)


@contextmanager
def progress_bar(total: int | None, desc: str, unit: str, leave: bool = False):
    if tqdm is None or not sys.stderr.isatty():
        progress = _NullProgressBar()
        try:
            yield progress
        finally:
            progress.close()
        return

    progress = tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        file=sys.stderr,
        mininterval=0.2,
    )
    try:
        yield progress
    finally:
        progress.close()
