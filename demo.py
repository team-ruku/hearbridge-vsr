import time

import hydra
from loguru import logger

from pipelines import InferencePipeline


@hydra.main(version_base="1.3", config_path="configs", config_name="hydra")
def main(cfg):
    logger.debug(f"Hydra config: {cfg}")

    if cfg.time:
        start = time.time()

    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.filename)
    print(f"transcript: {transcript}")

    if cfg.time:
        end = time.time()
        logger.debug(f"Exec time: {end-start}")


if __name__ == "__main__":
    main()
