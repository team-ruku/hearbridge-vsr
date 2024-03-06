import hydra
from pipelines import InferencePipeline


@hydra.main(version_base=None, config_path="configs", config_name="hydra")
def main(cfg):
    output = InferencePipeline(cfg.model, device_override=cfg.device)(
        cfg.data_filename, cfg.landmarks_filename
    )
    print(f"hyp: {output}")
