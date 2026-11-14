"""Read both original and worker-prefixed Stage2 file logs."""

from pathlib import Path
import re


_PREFIX = re.compile(r"^\[Worker #\d+\] \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ", re.MULTILINE)


def read_stage2_log(path: Path, *, encoding="utf-8-sig", errors="strict") -> str:
    return _PREFIX.sub("", path.read_text(encoding=encoding, errors=errors))
