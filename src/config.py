import os
from pathlib import Path

class Paths:
    # PROJECT ROOT
    ROOT = Path(__file__).resolve().parent.parent
    
    INPUT = ROOT / "input_csv_files"
    RESULTS = ROOT / "results"
    LOGS = ROOT / "logs"
    
    @classmethod
    def get_abs_path(cls, relative_path):
        """Converts a relative path in absolute respect to root."""
        return cls.ROOT / relative_path

    @classmethod
    def ensure_dir(cls, path):
        """Creates directory if not exists."""
        os.makedirs(path, exist_ok=True)
        return path

LOGS_DIR = Paths.LOGS

def ensure_dirs():
    """Inicializa todas las carpetas críticas del framework si no existen."""
    Paths.ensure_dir(Paths.INPUT)
    Paths.ensure_dir(Paths.RESULTS)
    Paths.ensure_dir(Paths.LOGS)