import time

import hydra
from loguru import logger

from pipelines import InferencePipeline


@hydra.main(version_base="1.3", config_path="configs", config_name="hydra")
def main(cfg):
    start = time.time()
    logger.debug(f"Hydra config: {cfg}")
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.filename)
    end = time.time()
    print(f"transcript: {transcript}")
    logger.debug(f"Exec time: {end-start}")


if __name__ == "__main__":
    main()
