import sys

import hydra
from loguru import logger

from pipelines import InferencePipeline


@hydra.main(version_base="1.3", config_path="configs", config_name="hydra")
def main(cfg):
    if not cfg.debug:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.debug(f"[Config] Hydra config: {cfg}")

    pipeline = InferencePipeline(cfg)


if __name__ == "__main__":
    main()
