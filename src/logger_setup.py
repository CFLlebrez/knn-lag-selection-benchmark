import logging
import sys
from .config import LOGS_DIR, ensure_dirs

def setup_framework_logger(name="FS-Framework"):
    ensure_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicates
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(LOGS_DIR / "framework.log")
        fh.setFormatter(formatter)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger