import hydra

from loguru import logger

from pipelines import InferencePipeline


@hydra.main(version_base="1.3", config_path="configs", config_name="hydra")
def main(cfg):
    logger.debug(f"Hydra config: {cfg}")

    model_config = {
        "model": {
            "v_fps": 25,
            "model_path": "models/visual/model.pth",
            "model_conf": "models/visual/model.json",
            "rnnlm": "models/language/model.pth",
            "rnnlm_conf": "models/language/model.json",
        },
        "decode": {
            "beam_size": 40,
            "penalty": 0.0,
            "maxlenratio": 0.0,
            "minlenratio": 0.0,
            "ctc_weight": 0.1,
            "lm_weight": 0.3,
        },
    }

    pipeline = InferencePipeline(cfg, model_config)
    transcript = pipeline(cfg.filename)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()
