# HearBridge VSR

## Getting Started

1. Clone the repository and enter it locally:

```Shell
git clone https://github.com/team-ruku/hearbridge-vsr
cd hearbridge-vsr
```

2. Setup the environment.

```Shell
conda create -y -n hearbridge-vsr python=3.8
conda activate hearbridge-vsr
```

3. Install pytorch, torchvision, and torchaudio by following instructions [here](https://pytorch.org/get-started/), and install all packages:

```Shell
pip3 install -r requirements.txt
conda install -c conda-forge ffmpeg
```

4. Download and extract a pre-trained model and language model from [download](https://bucket.2w.vc/public/hearbridge-vsr-models.zip) to:

- `./models/video/`

- `./models/audiovisual/`

- `./models/language/`

5. RUN

```Shell
python3 . data_filename=[data_filename]
```

### Miscellaneous

By default, HearBridge AVSR Module automatically matches torch device.

To override this, add `device` option.

```Shell
python3 . data_filename=[data_filename] device=cpu
```

To override model, use `model` option.

```Shell
python3 . data_filename=[data_filename] model=video device=cpu
```
