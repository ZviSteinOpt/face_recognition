import numpy as np
import torch
import torch.nn as nn
import glob
import cv2
import os
from torchvision.transforms import transforms
from captum.attr import IntegratedGradients
from PIL import Image
from torch.utils.data import DataLoader, random_split,WeightedRandomSampler
import torch.nn.functional as F

# Check available device: GPU > MPS > CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'

# Constants for data organization
TRUE_NAME = 'face'
FALSE_NAME = 'non-face'
TRAIN_NAME = 'train'
TEST_NAME = 'test'

class SimpleCNNWithBatchNorm(nn.Module):
    """A simple CNN with batch normalization for face recognition."""
    def __init__(self):
        super().__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(20 * 4 * 4, 5)
        self.bn_fc1 = nn.BatchNorm1d(5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x

class FaceRecognition:
    """Face recognition model that trains and evaluates a CNN with batch normalization."""
    def __init__(self, data_path,batch_size=10,num_epochs=50,lr=0.001,validation_split_factor=7):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.validation_split_factor = validation_split_factor

        self.create_augmented_data_loader_with_upsampling()
        self.train_model()
        self.evaluate_model()

    def train_model(self):

        self.model = SimpleCNNWithBatchNorm().to(device=device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        pos_weight = torch.tensor([1.5])

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_accuracy = 0.0
            train_loss = 0.0

            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                prediction = (torch.sigmoid(outputs) >= 0.5).float()
                train_accuracy += (prediction == labels).sum().item()

            train_accuracy /= self.train_count
            train_loss /= self.train_count
            validation_accuracy, validation_loss = self.evaluate_on_loader(self.validation_loader)

            print(f'Epoch: {epoch} Train Loss: {train_loss} Train Accuracy: {train_accuracy} '
                  f'Validation Loss: {validation_loss} Validation Accuracy: {validation_accuracy}')

    def evaluate_on_loader(self, loader):
        self.model.eval()
        total_accuracy = 0.0
        total_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()

        ground_truth = torch.tensor([], dtype=torch.float32)  # Initialize empty tensor for ground truths
        predictions = torch.tensor([], dtype=torch.float32)  # Initialize empty tensor for predictions

        with torch.no_grad():
            for images, labels in loader:
                outputs = self.model(images)
                loss = criterion(outputs, labels.float())
                total_loss += loss.item() * images.size(0)
                prediction = (torch.sigmoid(outputs) >= 0.5).float()
                total_accuracy += (prediction == labels).sum().item()
                ground_truth = torch.cat((ground_truth, labels.float()),
                                         dim=0)
                predictions = torch.cat((predictions, prediction),
                                        dim=0)
            ground_truth = ground_truth.cpu().numpy()
            predictions = predictions.cpu().numpy()

            tp = ((ground_truth == 1) & (predictions == 1)).sum()

            # True Negatives (TN): Both ground_truth and predictions are 0
            tn = ((ground_truth == 0) & (predictions == 0)).sum()

            # False Positives (FP): ground_truth is 0 but predictions are 1
            fp = ((ground_truth == 0) & (predictions == 1)).sum()

            # False Negatives (FN): ground_truth is 1 but predictions are 0
            fn = ((ground_truth == 1) & (predictions == 0)).sum()

        # print(
        #     f"True Positives: {tp / len(loader.dataset)}, True Negatives: {tn / len(loader.dataset)}, False Positives: {fp / len(loader.dataset)}, False Negatives: {fn / len(loader.dataset)}")
        print(f"Positives score: {tp / (fn + tp)}")
        total_accuracy /= len(loader.dataset)
        total_loss /= len(loader.dataset)
        return total_accuracy, total_loss

    def parse_data(self, folder_name):
        transformations = self.get_transformations(True if folder_name == TEST_NAME else False)

        data_container = []
        int_data_path = os.path.join(self.data_path, folder_name)
        for filename in glob.iglob(int_data_path + '/**/*.*', recursive=True):
            img = cv2.imread(filename, -1)
            label = np.array(1 if os.path.basename(os.path.dirname(filename)) == TRUE_NAME else 0).astype(np.float32)
            data_container.append([img, label[None,...]])

        dataset = [(transformation(Image.fromarray(img_label[0])).to(device=device), torch.tensor(img_label[1]).to(device=device))
                   for img_label in data_container for transformation in transformations]

        return dataset

    def create_augmented_data_loader(self):

        dataset = self.parse_data(TRAIN_NAME)

        self.validation_count = len(dataset) // self.validation_split_factor
        self.train_count = len(dataset) - self.validation_count

        train_dataset, validation_dataset = random_split(dataset, [self.train_count, self.validation_count])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

    def create_augmented_data_loader_with_upsampling(self):


        dataset = self.parse_data(TRAIN_NAME)


        self.validation_count = len(dataset) // self.validation_split_factor
        self.train_count = len(dataset) - self.validation_count

        train_dataset, validation_dataset = random_split(dataset, [self.train_count, self.validation_count])

        class_weights = [1, 2]  # class 0 has weight 1, class 1 has weight 5

        # Generate sample weights based on class weights and labels in the train dataset
        sample_weights = [class_weights[int(label.numpy())] for _, label in train_dataset]

        # # Ensure sample_weights is a tensor for compatibility with PyTorch DataLoader
        # sample_weights = torch.tensor(sample_weights, dtype=torch.float)


        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=self.train_count, replacement=True)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

    @staticmethod
    def get_transformations(for_test=False):
        norm_mean = [0.5]
        norm_std = [0.5]

        if not for_test:
            return [
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                    transforms.RandomGrayscale(0.4),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                    transforms.ColorJitter(brightness=0.5, hue=0.3),
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

    def evaluate_model(self):

        test_loader = DataLoader(self.parse_data(TEST_NAME), batch_size=self.batch_size, shuffle=True)
        test_accuracy, _ = self.evaluate_on_loader(test_loader)

        print(f'Test Accuracy: {test_accuracy}')

        # Integrated Gradients for a single batch
        images, _ = next(iter(test_loader))
        ig = IntegratedGradients(self.model)
        attributions, _ = ig.attribute(images, return_convergence_delta=True)

if __name__ == '__main__':
    # FaceRecognition('/path/to/your/data')
    FaceRecognition(r'/Users/zvistein/Documents/CV/work chalanges/faces')
