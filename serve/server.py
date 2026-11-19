"""Keep launchers generated before the 0.2.16 layout move working."""

from pathlib import Path
from runpy import run_path

if __name__ == "__main__":
    run_path(str(Path(__file__).resolve().parents[1] / "terminal" / "serve" / "server.py"),
             run_name="__main__")
