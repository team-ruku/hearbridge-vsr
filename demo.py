import sys

import hydra
import torch.multiprocessing as mp
import torch
import torchaudio
import torchvision
from loguru import logger

from pipelines import InferencePipeline
from pipelines.data import DataLoader


def stream(queue, format, segment_length):
    streamer = torchaudio.io.StreamReader(
        src="0",
        format=format,
        option={"framerate": "30", "pixel_format": "rgb24"},
    )
    streamer.add_basic_video_stream(
        frames_per_chunk=segment_length,
        buffer_chunk_size=500,
        width=600,
        height=340,
    )

    for chunk in streamer.stream(timeout=-1, backoff=1.0):
        queue.put(chunk)


@hydra.main(version_base="1.3", config_path="configs", config_name="hydra")
def main(cfg):
    if not cfg.debug:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    logger.debug(f"[Config] Hydra config: {cfg}")

    data_loader = DataLoader(cfg.filename, cfg.format)
    pipeline = InferencePipeline(cfg)
    ctx = mp.get_context("spawn")

    queue = ctx.Queue()

    @torch.inference_mode()
    def infer():
        num_video_frames = 0
        video_chunks = []

        while True:
            video_chunk = queue.get()[0]
            num_video_frames += video_chunk.size(0)

            video_chunks.append(video_chunk)

            if num_video_frames < cfg.buffer_size:
                continue

            video = data_loader.cacher(torch.cat(video_chunks))
            print(video)
            transcript = pipeline(video)
            print(transcript, end="", flush=True)

            num_video_frames = 0
            video_chunks = []

    process = ctx.Process(target=stream, args=(queue, cfg.format, cfg.segment_length))

    process.start()
    infer()
    process.join()


if __name__ == "__main__":
    main()
