import glob
import cv2
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split,WeightedRandomSampler
import os
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import torch.nn as nn
from captum.attr import IntegratedGradients

# Constants for data organization
TRUE_NAME = 'face'
FALSE_NAME = 'non-face'
TRAIN_NAME = 'train'
TEST_NAME = 'test'

class CNNClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, img_size=19):
        super().__init__()
        self.input_channels = input_channels
        self.img_size = img_size

        # Updated Convolutional layers setup with an additional layer
        input_num = 32
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, input_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(input_num, 2 * input_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * input_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2 * input_num, 3 * input_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3 * input_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dynamically calculate the size of the conv output to be fed into the FC layers
        self.flattened_size = self._get_conv_output_size(img_size)

        # Fully connected layers setup
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def _get_conv_output_size(self, img_size):
        # Simulate a forward pass through the conv layers using a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, img_size, img_size)
            dummy_output = self.conv_layers(dummy_input)
            return int(torch.numel(dummy_output) / dummy_output.shape[0])

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class DataParser:
    def __init__(self, data_path):
        """
        Initialize the DataParser instance.

        Parameters:
        - data_path (str): Path to the data directory.
        - device (torch.device): The device to use for tensor operations.
        """
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def parse_data(self, folder_name):
        """
        Parse data from a specified folder and apply transformations.

        Parameters:
        - folder_name (str): Name of the folder to parse data from.

        Returns:
        - List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples containing image tensors and labels.
        """
        is_test = folder_name == TEST_NAME
        transformations = self.get_transformations(is_test)

        data_container = self.load_data(folder_name)
        dataset = self.apply_transformations(data_container, transformations)

        return dataset

    def load_data(self, folder_name):
        """
        Load data and labels from a specified folder.

        Parameters:
        - folder_name (str): The folder name to load data from.

        Returns:
        - List[List[Any]]: A list of [image, label] pairs.
        """
        data_container = []
        int_data_path = os.path.join(self.data_path, folder_name)
        for filename in glob.iglob(int_data_path + '/**/*.*', recursive=True):
            img = cv2.imread(filename, -1)
            label = np.array(1 if os.path.basename(os.path.dirname(filename)) == TRUE_NAME else 0).astype(np.float32)
            data_container.append([img, label[None, ...]])

        return data_container

    def apply_transformations(self, data_container, transformations):
        """
        Apply transformations to the loaded data.

        Parameters:
        - data_container (List[List[Any]]): The loaded data.
        - transformations (List[Callable]): The transformations to apply.

        Returns:
        - List[Tuple[torch.Tensor, torch.Tensor]]: Transformed data as tensor pairs.
        """
        dataset = [
            (transformation(Image.fromarray(img_label[0])).to(self.device),
             torch.tensor(img_label[1]).to(self.device))
            for img_label in data_container for transformation in transformations
        ]

        return dataset

    def get_transformations(self, is_test):
        """
        Get the transformations to be applied to the data.

        Parameters:
        - is_test (bool): Flag indicating whether the transformations are for test data.

        Returns:
        - List[Callable]: A list of transformations.
        """
        norm_mean = [0.5]
        norm_std = [0.5]

        if not is_test:
            return [
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                    transforms.RandomRotation(degrees=(0, 35)),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                    transforms.ColorJitter(brightness=0.5, hue=0.3),
                    transforms.RandomRotation(degrees=(0, 35)),

                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                ])
            ]
        else:
            return [
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),])]

class ModelTrainer:
    """
    Trains and evaluates a face recognition CNN model.
    """
    def __init__(self, model, data_processor, config):
        self.model = model
        self.data_processor = data_processor
        self.config = config
        self.device = self.data_processor.device
        self.model.to(self.device)

    def train(self):
        """
        Train the model using the specified dataset and training configuration.
        """
        train_loader, validation_loader = self._prepare_data_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=0.15)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3])).to(self.device)

        for epoch in range(self.config['num_epochs']):
            self._train_one_epoch(train_loader, optimizer, criterion)
            self._validate(validation_loader)

        self._save_model()

    def _prepare_data_loaders(self):
        """
        Prepares training and validation data loaders with upsampling for the minority class.
        :return: A tuple of (train_loader, validation_loader)
        """
        dataset = self.data_processor.parse_data(TRAIN_NAME)
        validation_count = len(dataset) // self.config['validation_split_factor']
        train_count = len(dataset) - validation_count
        train_dataset, validation_dataset = random_split(dataset, [train_count, validation_count])

        sample_weights = self._get_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=train_count, replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], sampler=sampler)
        validation_loader = DataLoader(validation_dataset, batch_size=self.config['batch_size'], shuffle=True)

        return train_loader, validation_loader

    def _get_sample_weights(self, dataset):
        """
        Generates sample weights for upsampling based on class distribution.
        :param dataset: The training dataset.
        :return: A list of sample weights corresponding to each item in the dataset.
        """
        class_weights = [1, 2]  # Adjust based on class distribution
        sample_weights = [class_weights[int(label.cpu().numpy())] for _, label in dataset]
        return sample_weights

    def _train_one_epoch(self, train_loader, optimizer, criterion):
        """
        Trains the model for one epoch.
        :param epoch: The current epoch number.
        :param train_loader: DataLoader for the training data.
        :param optimizer: The optimizer.
        :param criterion: The loss function.
        """
        Performance = PerformanceMetrics(self.device)

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            prediction = torch.sigmoid(outputs) >= 0.5
            Performance.push(labels, prediction)
        Performance.calculate_accuracy()
        pass

    def _validate(self, validation_loader):
        """
        Validates the model on the validation dataset.
        :param epoch: The current epoch number.
        :param validation_loader: DataLoader for the validation data.
        """
        self.model.eval()
        performance = PerformanceMetrics(self.device)

        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = self.model(images)
                prediction = torch.sigmoid(outputs) >= 0.5
                performance.push(labels, prediction)
            performance.calculate_scores()

        pass

    def _save_model(self, save_path='model.pth'):
        """
        Saves the trained model to a specified path.

        Parameters:
        - model: The model to save.
        - save_path: Path to save the model file.
        """
        torch.save(self.model, save_path)

class PerformanceMetrics:
    """
    A class for computing and storing performance metrics for binary classification problems.
    """

    def __init__(self,device='cpu'):
        """
        Initializes the PerformanceMetrics class with placeholders for metrics.
        """
        self.device  = device
        self.tp = 0  # True Positives
        self.tn = 0  # True Negatives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives


        self.ground_truth = torch.tensor([], dtype=torch.float32).to(self.device)  # Initialize empty tensor for ground truths
        self.predictions = torch.tensor([], dtype=torch.float32).to(self.device)  # Initialize empty tensor for predictions

    def update_metrics(self):
        """
        Updates the performance metrics based on the provided ground truth and predictions.

        :param ground_truth: An array of ground truth labels.
        :param predictions: An array of predicted labels.
        """
        self.tp += ((self.ground_truth == 1) & (self.predictions == 1)).sum()
        self.tn += ((self.ground_truth == 0) & (self.predictions == 0)).sum()
        self.fp += ((self.ground_truth == 0) & (self.predictions == 1)).sum()
        self.fn += ((self.ground_truth == 1) & (self.predictions == 0)).sum()

    def calculate_scores(self):
        """
        Calculates and prints the positive score, negative score, and the percentage of TP, TN, FP, and FN.

        :return: A dictionary containing all calculated scores and percentages.
        """
        pos_score = self.tp / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0
        neg_score = self.tn / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
        total_predictions = self.tp + self.tn + self.fp + self.fn

        print(f"Positive score: {100 * pos_score:.2f}%")
        print(f"Negative score: {100 * neg_score:.2f}%")
        print(f"TP: {100 * self.tp / total_predictions if total_predictions > 0 else 0 :.2f}%")
        print(f"TN: {100 * self.tn / total_predictions if total_predictions > 0 else 0 :.2f}%")
        print(f"FP: {100 * self.fp / total_predictions if total_predictions > 0 else 0 :.2f}%")
        print(f"FN: {100 * self.fn / total_predictions if total_predictions > 0 else 0 :.2f}%")

    @staticmethod
    def extract_features(model, image_tensor):
        """
        Extracts and visualizes features from an image using the Integrated Gradients method.

        This function takes a pre-trained model and an image tensor as inputs, then computes
        the attributions of the input image with respect to the model's predictions. It
        visualizes the attributions as a heatmap overlay on the original image.

        Parameters:
        - model: torch.nn.Module
          The pre-trained model from which features are to be extracted.
        - image_tensor: torch.Tensor
          The input image tensor for which attributions are to be computed. Expected shape
          is [1, C, H, W], where C is the number of channels, and H, W are the height and
          width of the image.

        Returns:
        None. The function visualizes the attributions heatmap overlay directly.
        """
        import matplotlib.pyplot as plt
        model.eval()

        # Initialize Integrated Gradients with the provided model
        ig = IntegratedGradients(model)

        # Compute attributions using Integrated Gradients
        # Note: No target index is specified, assuming a single output from the model
        attributions, delta = ig.attribute(image_tensor, return_convergence_delta=True)

        # Process attributions for visualization
        attributions_np = attributions.detach().numpy()[0, 0, ...]
        input_image_np = image_tensor.detach().numpy()[0, 0, ...]

        # Normalize attributions and input image for better visualization
        attributions_normalized = (attributions_np - np.min(attributions_np)) / np.ptp(attributions_np)
        input_image_normalized = (input_image_np - np.min(input_image_np)) / np.ptp(input_image_np)

        # Create visualization
        fig, ax = plt.subplots()
        ax.imshow(input_image_normalized, cmap='gray', interpolation='nearest')
        heatmap = ax.imshow(attributions_normalized, cmap='jet', alpha=0.3, interpolation='nearest')
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('Attribution Strength')
        ax.set_title("Attributions Heatmap Overlay")
        plt.axis('off')  # Hide axis for a cleaner visualization

    def calculate_accuracy(self):
        """
        Calculates the total accuracy.

        :param total_accuracy: The summed accuracy over all batches.
        :param loader: The data loader containing the dataset.
        :return: The total accuracy normalized by the dataset size.
        """
        pos_score = self.tp / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0
        neg_score = self.tn / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
        accuracy = 0.5*pos_score+0.5*neg_score
        print(f"Accuracy: {100 * accuracy:.2f}%")

    def push(self, ground_truth, prediction):
        self.ground_truth = torch.cat((self.ground_truth, ground_truth.float()),
                                 dim=0)
        self.predictions = torch.cat((self.predictions, prediction),
                                dim=0)
        self.update_metrics()

class ModelTester:
    """
    A class for testing face recognition models. It supports loading models, combining predictions from two models,
    and evaluating the combined model on a test dataset.
    """

    def __init__(self, data_processor, model_paths):
        """
        Initializes the FaceModelTester with a test DataLoader and model paths.

        Parameters:
        - test_data_loader: A DataLoader containing the test dataset.
        - test_save_path: Path where test results and images are saved.
        - model_paths: A tuple or list containing paths to the two models to be combined for testing.
        """
        self.device = data_processor.device
        self.test_data_loader = DataLoader(data_processor.parse_data(TEST_NAME), batch_size=1, shuffle=True)
        self.test_save_path = os.path.join(data_processor.data_path,'res')
        os.makedirs(self.test_save_path, exist_ok=True)
        self.model_paths = model_paths
        self.models = [self.load_model(path) for path in model_paths]

    def load_model(self, model_path):
        """
        Loads a model from a given file path.

        Parameters:
        - model_path: Path to the model file.

        Returns:
        - The loaded model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = torch.load(model_path)
        model.eval()
        return model

    def test_models(self):
        """
        Tests the loaded models on the test dataset, combining predictions where necessary,
        and saves output images for analysis. Prints evaluation metrics.
        """
        if not os.path.exists(self.test_save_path):
            os.makedirs(self.test_save_path)
        performance = PerformanceMetrics(self.device)
        # Directories for true and false predictions
        true_path = os.path.join(self.test_save_path, TRUE_NAME)
        false_path = os.path.join(self.test_save_path, FALSE_NAME)
        os.makedirs(true_path, exist_ok=True)
        os.makedirs(false_path, exist_ok=True)


        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_data_loader):
                outputs = self.models[0](images)
                prediction = torch.sigmoid(outputs) >= 0.5

                # Use the second model if the first model's prediction is uncertain
                if len(self.models)>1 and prediction.sum() == 0:
                    outputs = self.models[1](images)
                    prediction = torch.sigmoid(outputs) >= 0.5

                performance.push(labels, prediction)

                self.save_output_images(i,images,labels, prediction, true_path, false_path)
            performance.calculate_scores()


    def save_output_images(self,idx, images,label,predictions, true_path, false_path):
        """
        Saves output images in separate directories based on prediction results.

        Parameters:
        - images: Tensor of images from the DataLoader.
        - predictions: Tensor of model predictions.
        - true_path: Path to save images for positive label.
        - false_path: Path to save images for negative label.
        """
        for i, image in enumerate(images):
            prediction = predictions[i]
            img = F.to_pil_image(image)
            if label:
                img.save(os.path.join(true_path, f"{(i + 1) * (idx + 1)}_{prediction.cpu().numpy()}.png"))
            else:
                img.save(os.path.join(false_path, f"{(i + 1) * (idx + 1)}_{prediction.cpu().numpy()}.png"))

def main():
    """
    Main function to instantiate model, data processor, and trainer classes and start the training process.
    """
    data_path = r'/path/to/data/'

    config = {
        'data_path': data_path,
        'batch_size': 10,
        'num_epochs': 1,
        'lr': 0.0001,
        'validation_split_factor': 7
    }

    data_processor = DataParser(data_path)

    model = CNNClassifier()
    trainer = ModelTrainer(model, data_processor, config)
    trainer.train()
    tester = ModelTester(data_processor, ['model_0.87_0.89.pth','model_0.72_0.94.pth'])
    tester.test_models()

if __name__ == '__main__':
    main()
