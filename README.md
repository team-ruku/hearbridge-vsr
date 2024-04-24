# HearBridge Visual Speech Recognition

> Conversation Textualization via Lip-reading, supporting multi-party conversations

## Prerequisites

1. Setup the conda environment.

```bash
conda create -y -n hearbridge python=3.8
conda activate hearbridge
```

2. Install requirements.

```bash
python3 -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

3. Download & Extract pre-trained models to:

- `models/visual/model.pth`
- `models/spm`
- `models/mediapipe/short_range.tflite`
- `models/mediapipe/face_landmarker.tflite`

## Run

```bash
python3 demo.py
```

- For debugging, add `debug` flag.

## Reference

```bibtex
@inproceedings{ma2023auto,
  author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096889}
}
```
