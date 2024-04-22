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

## Run

```bash
python3 demo.py filename=[video file]
```

In the real-time case, you should write `avfoundation` or camera index number on the `filename`.

### Misc

- If you want to check the execution time, add `time` flag.

```bash
python3 demo.py filename=[video file] time=true
```

- If you want to check the mouth ROI crop result, add `save_mouth_roi` flag.

```bash
python3 demo.py filename=[video file] save_mouth_roi=true
```

- If you want to debug the instance, add `debug` flag.

```bash
python3 demo.py filename=[video file] debug=true
```

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
