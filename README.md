# Violence Detection

A deep-learning project that detects violence in videos using a **CNN + LSTM** architecture. The project is built with **TensorFlow / Keras** and **OpenCV**, and consists of two Jupyter notebooks:

1. `violence_détection .ipynb` — trains and evaluates a violence/non-violence classifier.
2. `violence_détection_real_time.ipynb` — performs real-time (webcam / recorded video) violence detection with on-screen alerts.

## How it works

The pipeline uses a **transfer-learning** approach:

1. **Frame extraction** — each video is sampled into a fixed number of frames (default `20`, resized to `224×224` RGB).
2. **Feature extraction** — frames are passed through a pre-trained **VGG16** model (ImageNet weights); the output of the `fc2` layer (a `4096`-dimensional feature vector) is used as the frame representation.
3. **Sequence classification** — the sequence of frame features is fed into an **LSTM** network that classifies a video as *Violent* or *Non-Violent*.

### Model architecture

```
LSTM (512 units)
-> Dense (1024, ReLU)
-> Dense (50, Sigmoid)
-> Dense (2, Softmax)   # [Violence, Non-Violence]
```

## Notebooks

### `violence_détection .ipynb` (Training & Evaluation)

- Reads video files named `F*.mp4` (Violence) and `N*.mp4` (Non-Violence).
- Extracts VGG16 transfer values for each video.
- Stores the preprocessed features in HDF5 files (`prueba2.h5` for training, `pruebavalidation2.h5` for validation).
- Splits data into training (80%) and validation (20%) sets.
- Trains the LSTM classifier (with early stopping) and reports accuracy / loss.
- Saves the trained model to `modelVGG16.h5`.
- Includes an `infer()` helper to classify a single video file.

### `violence_détection_real_time.ipynb` (Real-Time Inference)

- Loads the pre-trained LSTM model (`modelVGG16.h5`) and VGG16 for feature extraction.
- Reads a video stream (set `input_path = 0` for webcam, or a video file path).
- Processes frames in 1-second windows (30 frames at 30 fps), extracts features, and predicts violence.
- Averages predictions over the last 5 seconds for stability.
- Overlays a live caption on the video:
  - **Green "Normal"** when no violence is detected.
  - **Red "Violence Alert!"** with a confidence percentage when violence is detected.
- Saves the annotated video to `output04.mp4` and shows it in a preview window (press `ESC` to quit).

## Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV (`cv2`)
- NumPy
- scikit-image (`skimage`)
- Pillow (`PIL`)
- h5py
- matplotlib
- Jupyter Notebook


## Usage

1. **Training**
   - Update `in_dir` in the training notebook to point to your dataset folder.
   - Run all cells to extract features, train the LSTM, and save `modelVGG16.h5`.

2. **Real-Time Detection**
   - Place `modelVGG16.h5` in the working directory.
   - Set `input_path` to `0` (webcam) or a video file.
   - Run all cells in the real-time notebook.

## Dataset

The classifier expects videos named with the following convention:

- `F*.mp4` — videos containing **violence** (label `[1, 0]`).
- `N*.mp4` — **non-violent** videos (label `[0, 1]`).

