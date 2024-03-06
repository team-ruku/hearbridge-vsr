import torch
import hydra
from pipelines import InferencePipeline


@hydra.main(version_base=None, config_path="configs", config_name="hydra")
def main(cfg):
    device = torch.device(
        f"cuda:{cfg.gpu_idx}"
        if torch.cuda.is_available() and cfg.gpu_idx >= 0
        else "cpu"
    )
    output = InferencePipeline(cfg.config_filename, device=device)(
        cfg.data_filename, cfg.landmarks_filename
    )
    print(f"hyp: {output}")


if __name__ == "__main__":
    main()
