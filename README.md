# Face Recognition Model

This repository contains a face recognition model that utilizes Convolutional Neural Networks (CNNs) for the identification and verification of individuals in images. The project is structured into two main stages: training and testing.

## Getting Started

### Prerequisites

- Python 3.8 or newer
- PyTorch 1.8.1 or newer
- OpenCV
- torchvision
- NumPy
- Pillow
- Captum

It's recommended to use a virtual environment to manage dependencies.

### Installation

1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/ZviSteinOpt/face_recognition.git
   ```
2. Navigate to the repository folder and create a virtual environment.
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. Install the required dependencies.
   ```bash
   pip install torch torchvision numpy opencv-python pillow captum
   ```

### Data Preparation

Download the dataset from the following link and extract it into a directory named `data` in the root of this repository.

[Download Faces Dataset](http://www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz)

```bash
wget http://www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz
tar -xzvf faces.tar.gz -C ./data
```

## Training the Model

To train the model, run the following command:

```bash
python FaceClassifier.py --mode train --data_path "/path/to/data" --num_epochs 10 --batch_size 10 --lr 0.0001
```

This will initiate the training process on the dataset located at `./data`. The trained CNN model will be saved for future use.

## Testing the Model

For testing the pre-trained model or any other specified model, use the following command:

```bash
python FaceClassifier.py --mode test --data_path "/path/to/data" --model_path model_0.87_0.89.pth model_0.72_0.94.pth
```

If no model path is specified, the script will automatically use the pre-trained model included in the repository.

## Loading and Using the Trained Model

To load and use the saved model, follow these steps:

1. Import the CNN model architecture (`CNNClassifier`) from the model definition file.
2. Load the model using PyTorch's `load` function with the appropriate model path.

Example:

```python
import torch
from model import CNNClassifier

model = torch.load('path_to_your_saved_model.pth')
model.eval()  # Set the model to evaluation mode
```
