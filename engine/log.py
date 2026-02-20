import logging

try:
    import colorama
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False

ROOT_TAG = "comic-separator"


class _ColorFormatter(logging.Formatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        if _HAS_COLORAMA and record.levelno >= logging.ERROR:
            self._style._fmt = (
                f"{colorama.Fore.RED}%(levelname)s:{colorama.Fore.RESET} "
                "[%(name)s] %(message)s"
            )
        elif _HAS_COLORAMA and record.levelno >= logging.WARNING:
            self._style._fmt = (
                f"{colorama.Fore.YELLOW}%(levelname)s:{colorama.Fore.RESET} "
                "[%(name)s] %(message)s"
            )
        else:
            self._style._fmt = "[%(name)s] %(message)s"
        return super().formatMessage(record)


class _TagFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith(ROOT_TAG):
            return False
        prefix = ROOT_TAG + "."
        if record.name.startswith(prefix):
            record.name = record.name[len(prefix):]
        return True


_root = logging.getLogger(ROOT_TAG)


def setup_logging(level: int = logging.INFO) -> None:
    """Call once at startup to attach coloured handler."""
    logging.basicConfig(level=level)
    for h in logging.root.handlers:
        h.setFormatter(_ColorFormatter())
        h.addFilter(_TagFilter())


def get_logger(name: str) -> logging.Logger:
    return _root.getChild(name)
