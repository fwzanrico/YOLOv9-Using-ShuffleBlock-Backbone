
# YOLOv9 with Shuffle Block Backbone

This repository contains a modified implementation of [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://github.com/ultralytics/yolov9) that integrates a ShuffleNet-based backbone to improve performance and efficiency. This approach leverages the lightweight architecture of ShuffleNet to enhance YOLOv9 for real-time applications.

### Base Code Attribution
This repository uses the YOLOv9 codebase from the [YOLOv9 GitHub Repository](https://github.com/ultralytics/yolov9) as a foundation. We have modified this code to incorporate a ShuffleNet backbone for a more lightweight and efficient architecture suited for real-time applications.

## Installation

### Step 1: Install PyTorch
Before starting, ensure you have the appropriate version of PyTorch installed based on your CUDA version.

1. Check your CUDA version by running the following command:
   ```bash
   nvidia-smi
   ```
2. Install PyTorch, `torchvision`, and `torchaudio` with the matching CUDA version. Replace `cuXXX` with your CUDA version (e.g., `cu116` for CUDA 11.6):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
   ```

### Step 2: Install Required Dependencies
Install the remaining dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Set Up Your Dataset
Ensure your dataset is formatted correctly and that the `data.yaml` file specifies the correct paths to your dataset folders.

## Training the Model
To train the YOLOv9 model with the ShuffleNet backbone, use the following command:

```bash
python train_dual.py --batch 16 --epochs 300 --data dataset/data.yaml --cfg models/detect/yolov9tori.yaml --device 0 --patience 10 --save-period 1 --project runs/train --name yolov9_ori
```

#### Arguments
- `--batch`: Batch size (e.g., 16).
- `--epochs`: Number of training epochs (e.g., 300).
- `--data`: Path to the dataset configuration YAML file (e.g., `dataset/data.yaml`).
- `--cfg`: Path to the model configuration file (e.g., `models/detect/yolov9tori.yaml`).
- `--device`: GPU device number (e.g., `0` for the first GPU). If you don’t have a GPU, omit this option to run on CPU.
- `--patience`: Number of epochs with no improvement before early stopping (e.g., `10`).
- `--save-period`: Interval (in epochs) to save model checkpoints (e.g., `1`).
- `--project`: Folder to save training results (e.g., `runs/train`).
- `--name`: Name of the training run (e.g., `yolov9_ori`).

## Running the Model
After training, you can use the model to make predictions:

```bash
python detect_dual.py --weights yoloxshuffle.pt --source 0 --device 0
```

#### Arguments
- `--weights`: Path to the trained model weights (e.g., `yoloxshuffle.pt`).
- `--source`: Input source for predictions. It could be a video file, image file, or camera (`0` for webcam input).
- `--device`: Device to run the model on. Set to `0` for GPU. If using a CPU, omit this option.

**Note**: If you don’t have a CUDA-capable GPU, remove the `--device` argument to run on the CPU.

## Repository Structure
- `models/`: Contains YOLOv9 model configurations.
- `dataset/`: Folder for the dataset, with `data.yaml` for configuration.
- `runs/`: Directory where training results are saved.
- `train_dual.py`: Script for training the YOLOv9 model with ShuffleNet backbone.
- `detect_dual.py`: Script for running inference with the trained model.

## License
This repository follows the license guidelines of the original YOLOv9 project. For more details, refer to the [YOLOv9 License](https://github.com/ultralytics/yolov9).
