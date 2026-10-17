# Simpsons Character Classifier

A small convolutional neural network (CNN), built with PyTorch, that classifies images of characters from *The Simpsons*. It trains on a folder of labelled images and then predicts a character name for every image in a separate test folder.

## How it works

- **Data:** images are loaded with `torchvision.datasets.ImageFolder`, so each character is a sub-folder of `data/characters_train/` and the folder name is the class label.
- **Preprocessing:** images are resized to 64×64 and normalized to the range [-1, 1]. Training adds light augmentation (random horizontal flip, rotation up to 10°, and brightness/contrast/saturation jitter).
- **Split:** 15% of the training images are held out for validation (fixed seed, so the split is reproducible).
- **Model (`SimpleCNN`):** four convolutional blocks (3→32→64→128→256 channels, each with BatchNorm and ReLU, the first three followed by 2×2 max-pooling and the last by adaptive average pooling to 4×4), followed by a classifier head: Linear(4096→512), ReLU, Dropout(0.4), Linear(512→number of classes).
- **Training:** Adam optimizer (learning rate 1e-3, weight decay 1e-4), cross-entropy loss, batch size 32, 25 epochs. The learning rate is halved when validation loss stops improving for 3 epochs. The checkpoint with the lowest validation loss is saved.
- **Inference:** the saved checkpoint is loaded, every `.jpg`, `.jpeg` and `.png` file in the test folder is classified, and the predictions are written to `results.json` as `{"filename": "predicted_class"}`.

## Project structure

```
simpsons_project/
├── data/
│   ├── characters_train/   # training images, one sub-folder per character
│   └── test/               # unlabelled images to classify
├── train.py                # trains the model, saves model.pth and training_curves.png
├── train.ipynb             # notebook version of the training script
├── inference.py            # runs the model on data/test, writes results.json
├── inference.ipynb         # notebook version of the inference script
├── model.pth               # trained checkpoint (weights, class names, image size)
├── results.json            # predictions produced by inference
├── training_curves.png     # loss and accuracy curves from training
└── .gitignore
```

## Requirements

- Python 3.8+
- `torch` and `torchvision`
- `numpy`
- `matplotlib`
- `Pillow`

```bash
pip install torch torchvision numpy matplotlib pillow
```

A GPU is used automatically if available; the model is small enough to train on CPU.

## Usage

### 1. Prepare the data

Arrange the training images with one folder per character:

```
data/characters_train/
├── bart_simpson/
│   ├── img001.jpg
│   └── ...
├── homer_simpson/
│   └── ...
└── ...
```

Put the images you want to classify in `data/test/` (`.jpg`, `.jpeg` or `.png`).

### 2. Train

```bash
python train.py
```

This prints the epoch-by-epoch loss and accuracy, saves the best checkpoint to `model.pth`, and writes `training_curves.png`.

### 3. Run inference

```bash
python inference.py
```

This classifies every image in `data/test/` and saves the predictions to `results.json`, for example:

```json
{
    "image_01.jpg": "bart_simpson",
    "image_02.jpg": "homer_simpson"
}
```

The first 10 predictions are also printed to the console. Images that cannot be read are recorded as `"unknown"`.

## Configuration

Hyperparameters and paths are constants at the top of `train.py`:

| Setting | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `data/characters_train` | Training data folder |
| `MODEL_PATH` | `model.pth` | Where the checkpoint is saved |
| `IMG_SIZE` | `64` | Images are resized to this square size |
| `BATCH_SIZE` | `32` | Batch size |
| `EPOCHS` | `25` | Number of training epochs |
| `LR` | `1e-3` | Initial learning rate |
| `VAL_SPLIT` | `0.15` | Fraction of data held out for validation |
| `SEED` | `42` | Random seed for reproducibility |

The test folder and model path for inference are set at the bottom of `inference.py`.

## Notes

- `inference.py` redefines `SimpleCNN` so it can run on its own. If you change the architecture in `train.py`, make the same change in `inference.py`, or the saved weights will not load.
- The class names and image size are stored inside `model.pth`, so inference does not need the training data.
