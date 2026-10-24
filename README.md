# FashionNet: Virtual Try-On
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
<<<<
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![YAML](https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white)
FashionNet is a virtual try-on project built around a clothes segmentation model. Given an input image of a person, the model segments the clothing region and produces a mask that can be used in downstream virtual try-on pipelines. The repository provides a ready-to-use pre-built model for quick inference, along with a complete workflow for preparing a dataset and training a custom model.
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
## Table of Contents

1. [Technology Stack](#technology-stack)
2. [Repository Branches](#repository-branches)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Quick Start: Run the Pre-Built Model](#quick-start-run-the-pre-built-model)
6. [Training a Custom Model](#training-a-custom-model)
   - [Step 1: Prepare the Dataset](#step-1-prepare-the-dataset)
   - [Step 2: Train the Model](#step-2-train-the-model)
   - [Step 3: Run Inference](#step-3-run-inference)
7. [GPU and CPU Versions](#gpu-and-cpu-versions)
8. [Requirements](#requirements)
9. [Troubleshooting](#troubleshooting)

---

## Technology Stack

| Technology | Logo | Purpose |
|------------|------|---------|
| Python | <img src="https://cdn.simpleicons.org/python" alt="Python" height="32"> | Core programming language for the entire project |
| PyTorch | <img src="https://cdn.simpleicons.org/pytorch" alt="PyTorch" height="32"> | Model definition, training, and inference |
| OpenCV | <img src="https://cdn.simpleicons.org/opencv" alt="OpenCV" height="32"> | Image reading, resizing, and mask processing |
| MediaPipe | <img src="https://cdn.simpleicons.org/google" alt="MediaPipe" height="32"> | Pose and landmark detection for person-aware image processing |
| NumPy | <img src="https://cdn.simpleicons.org/numpy" alt="NumPy" height="32"> | Numerical operations on image arrays |
| Pandas | <img src="https://cdn.simpleicons.org/pandas" alt="Pandas" height="32"> | Reading and handling CSV annotation and label files |
| CUDA | <img src="https://cdn.simpleicons.org/nvidia" alt="CUDA" height="32"> | GPU acceleration for training and inference (GPU version) |
| YAML | <img src="https://cdn.simpleicons.org/yaml" alt="YAML" height="32"> | Configuration files for training and inference settings |

---

## Repository Branches

| Branch             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `main`             | Primary branch for systems with a GPU. Contains the pre-built model code.   |
| `main-cpu`         | CPU-only version for systems without a GPU.                                 |
| `image-processing` | Dataset preparation scripts used to generate image and mask pairs.          |

The commands are the same for the GPU and CPU versions. The differences lie in the YAML configuration files and a few hardware-specific settings.

---

## Project Structure

```
FashionNet/
|-- cloths_segmentation/
|   |-- run.py            # Quick inference using the pre-built model
|   |-- train.py          # Model training entry point
|   |-- inference.py      # Batch inference with custom weights
|   |-- ...
|-- ImageProcessing/      # Available on the image-processing branch
|   |-- dataset.py        # Dataset preparation (image and mask separation)
|-- requirements.txt
|-- README.md
```

The structure above is indicative. Refer to the repository for the exact file layout of each branch.

---

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd FashionNet
   ```

2. Select the branch that matches your hardware:

   ```bash
   # System with GPU
   git checkout main

   # System without GPU
   git checkout main-cpu
   ```

3. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv

   # Linux / macOS
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If a `requirements.txt` file is not available, install the core libraries manually:

   ```bash
   pip install numpy pandas opencv-python mediapipe torch torchvision
   ```

---

## Quick Start: Run the Pre-Built Model

Use this option to run the existing model on an input image without any training.

1. Clone the repository and check out the correct branch (see [Installation](#installation)).
2. Place your input image in the location expected by the script.
3. Run the following command:

   ```bash
   python cloths_segements/run.py
   ```

The script processes the given input image and displays or saves the output result.

---

## Training a Custom Model

To train your own model, you first need image and mask pairs. The dataset preparation code is located on a separate branch.

### Step 1: Prepare the Dataset

1. Download a clothing segmentation dataset from Kaggle.
2. Switch to the image processing branch:

   ```bash
   git checkout image-processing
   ```

3. Run the dataset preparation script. Make sure all paths are correct before running the command.Before run give path in `run_dataset.sh` file then run commad 
  ```bash
      bash run_dataset.sh
  ```

   | Argument | Description                                               |
   |----------|-----------------------------------------------------------|
   | `-i`     | Path to the downloaded dataset images                     |
   | `-l`     | Path to the CSV file containing the labels                |
   | `-o`     | Output directory for the processed images and masks       |

The script separates the original images from their corresponding mask images based on the dataset. The output directory will contain the data required for training.

### Step 2: Train the Model

Return to the main branch that matches your hardware (`main` for GPU, `main-cpu` for CPU), then start training with your prepared data:

```bash
python -m cloths_segmentation.train -c <path to config>
```

| Argument | Description                                       |
|----------|---------------------------------------------------|
| `-c`     | Path to the YAML configuration file for training  |

Update the dataset paths, batch size, number of epochs, and other settings in the configuration file before training.

### Step 3: Run Inference

After training, run inference on new images using the trained weights:

```bash
python -m torch.distributed.launch --nproc_per_node=<num_gpu> cloths_segmentation/inference.py \
    -i <path to images> \
    -c <path to config> \
    -w <path to weights> \
    -o <output path> \
    --fp16
```

| Argument            | Description                                           |
|---------------------|-------------------------------------------------------|
| `--nproc_per_node`  | Number of GPUs to use                                 |
| `-i`                | Path to the input images                              |
| `-c`                | Path to the configuration file                        |
| `-w`                | Path to the trained model weights                     |
| `-o`                | Directory where the output masks will be saved        |
| `--fp16`            | Enables half-precision inference (GPU only)           |

---

## GPU and CPU Versions

| Aspect              | `main` (GPU)                         | `main-cpu` (CPU)                       |
|---------------------|--------------------------------------|----------------------------------------|
| Hardware            | NVIDIA GPU with CUDA                 | Any system without a GPU               |
| Commands            | Same as listed in this document      | Same as listed in this document        |
| YAML configuration  | GPU-specific settings                | CPU-specific settings                  |
| Performance         | Faster training and inference        | Slower, suitable for testing and demos |

Notes for CPU users:

- Use the configuration files provided on the `main-cpu` branch.
- Half-precision inference (`--fp16`) and multi-GPU launch options apply only to GPU systems. Adjust the inference command according to the instructions on the `main-cpu` branch.

---

## Requirements

- Python 3.8 or later
- PyTorch (CUDA-enabled build recommended for the GPU version)
- OpenCV
- MediaPipe
- NumPy
- Pandas
- NVIDIA GPU with CUDA support (GPU version only)
- A clothing segmentation dataset from Kaggle (required only for custom training)

---

## Troubleshooting

| Issue                                   | Suggested Solution                                                              |
|-----------------------------------------|---------------------------------------------------------------------------------|
| `ModuleNotFoundError`                   | Ensure the virtual environment is active and all dependencies are installed.    |
| File or path not found                  | Verify that the dataset, CSV, config, and weights paths are correct.            |
| CUDA out of memory                      | Reduce the batch size in the configuration file or use the CPU branch.          |
| Script not found on current branch      | Confirm you are on the correct branch (`main`, `main-cpu`, or `image-processing`). |

---