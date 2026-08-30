import logging
import os

logging.basicConfig(
    level=os.getenv("PRODRAG_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
