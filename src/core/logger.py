import logging
import logging.config
import yaml
import os

def setup_logging():
    config_path = "config/logging.yaml"
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

setup_logging()
logger = logging.getLogger("mini_llm")