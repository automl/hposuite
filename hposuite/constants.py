from __future__ import annotations

from pathlib import Path

DEFAULT_STUDY_DIR = Path.cwd().absolute().parent / "hposuite-output"
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"