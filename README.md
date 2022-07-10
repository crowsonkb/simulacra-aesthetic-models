## Introduction

This is model fit and inference code for CLIP aesthetic regressions trained on
[Simulacra Aesthetic Captions](https://github.com/JD-P/simulacra-aesthetic-captions).
These remarkably simple models emulate human aesthetic judgment. They can be used
in tasks such as dataset filtering to remove obviously poor quality images from
the corpus before training. The following grids, one sorted by [John David
Pressman](https://github.com/JD-P) and one sorted by the machine give some idea
of the models capabilities:

### Manually Sorted Grid

![A human sorted grid of 20 images from worst to best, starting with the worst image in the
top left and the best in the bottom right](https://github.com/crowsonkb/simulacra-aesthetic-models/raw/master/sacManualSort.png)

### Model Sorted Grid

![A machine sorted grid of 20 images from worst to best, starting with the worst image in the
top left and the best in the bottom right](https://github.com/crowsonkb/simulacra-aesthetic-models/raw/master/sacModelSort.png)

## Installation

Git clone this repository:

```
git clone https://github.com/crowsonkb/simulacra-aesthetic-models.git
```

Install pytorch if you don't already have it:

```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Then pip install our other dependencies:

```
pip3 install tqdm pillow torchvision sklearn numpy
```

If you don't already have it installed, you'll need to install CLIP:

```
git clone https://github.com/openai/CLIP.git
cd CLIP
pip3 install .
cd ..
```

## Usage

The models are largely meant to be used as a library, i.e. you'll need to write
specific code for your use case. But to get you started we've provided a sample
script `rank_images.py` which finds all the `.jpg` or `.png` images in a directory
tree and ranks the top N (default 50) with the aesthetic model:

```
python3 rank_images.py demo_images/
```
