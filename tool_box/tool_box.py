from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# TODO: add comments
# TODO: write docstrings
# TODO: add proper type annotations

class ImageDataSetTemplate(Dataset):
    """
    Dataset template for image classification.

    Args:
        data_dir: str
        transform: torchvision.transforms
    """

    def __init__(
        self,
        data_dir: str,
        transform,
    ) -> None:
        self.data = ImageFolder(data_dir, transform=transform)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_classes(self) -> int:
        return len(self.data.classes)

    @property
    def classes(self) -> dict[int, str]:
        return {v: k for k, v in self.data.class_to_idx.items()}

    @property
    def transform(self):
        return self.data.transform


class ModelTemplate(nn.Module):
    """
    Model template

    Args:
        model: nn.Module        Takes in a base model
        num_classes: int        Number of classes to be trained on
        enet_out_size: int      Output size of the last layer in the base model
        classifier: nn.Module   New classifier layer equal to num_classes
    """

    def __init__(self, model, classifier):
        super(ModelTemplate, self).__init__()
        self.base_model = model
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class Trainer:
    """
    Trainer class for training and validation

    Args:
        model: nn.Module            Model to be trained
        optimizer: optim.Optimizer  Optimizer to be used
        criterion                   Criterion to be used
        device: str                 Device to be used
    """

    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion, device: str
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_loop(
        self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int
    ) -> None:
        """
        Train model loop

        Args:
            train_loader: DataLoader    Training dataset
            val_loader: DataLoader      Validation dataset
            num_epochs: int             Number of epochs to train
        """
        # Train model loop start
        for epoch in range(num_epochs):
            # Set model to training mode
            self.model.train()
            # Send model to device
            self.model.to(self.device)
            # Initialize running loss
            running_loss = 0.0
            # Train model
            for images, labels in tqdm(train_loader, desc="Training.."):
                # Send data to device
                images, labels = images.to(self.device), labels.to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward + Backward + Optimize
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # Update running loss
                running_loss += loss.item() * labels.size(0)
            # Calculate train loss
            train_loss = running_loss / len(train_loader)
            # Append train loss to list
            self.train_losses.append(train_loss)
            # Validation
            # call _validate for validation loss
            self._validate(val_loader)
            # Print results
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(
                f"Train loss: {self.train_losses[epoch]}, Val loss: {self.val_losses[epoch]}"
            )

    def _validate(self, val_loader: DataLoader):
        """
        Validation loop called from train_loop

        Args:
            val_loader: DataLoader    Validation dataset
        """
        # Set model to evaluation mode
        self.model.eval()
        # Send model to device
        self.model.to(self.device)
        # Initialize running loss
        running_loss = 0.0
        # Set model to evaluation mode
        with torch.no_grad():
            # Validation
            for images, labels in val_loader:
                # Send data to device
                images, labels = images.to(self.device), labels.to(self.device)
                # Forward
                outputs = self.model(images)
                # Calculate loss
                loss = self.criterion(outputs, labels)
                # Update running loss
                running_loss += loss.item() * labels.size(0)
                # Calculate validation loss
                val_loss = running_loss / len(val_loader)
        # Append validation loss to list
        val_loss = running_loss / len(val_loader)
        self.val_losses.append(val_loss)

    def plot_loss(self) -> None:
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


class Evaluater:
    def __init__(self, model, dataset, device) -> None:
        self.model = model
        self.device = device
        self.dataset = dataset

    def evaluate(self, num_images=10):
        # Load test images
        test_images = glob("dataset/test/*/*")
        test_examples = np.random.choice(test_images, num_images)

        for example in test_examples:
            # Preprocess image
            original_image, image_tensor = self._preprocess_image(
                example, self.dataset.transform
            )
            # Predict
            probabilities = self._predict(image_tensor)
            # Assuming dataset.classes gives the class names
            class_names = self.dataset.classes.values()
            self.visualize_predictions(original_image, probabilities, class_names)

    def accuracy(self, val_loader):
        total_correct = 0
        total_instances = 0
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                classifications = torch.argmax(self.model(images), dim=1)
                correct_predictions = torch.sum(classifications == labels)
                total_correct += correct_predictions
                total_instances += len(images)
        return float(total_correct / total_instances)

    def _preprocess_image(self, image_path, transform):
        # Load image
        image = Image.open(image_path).convert("RGB")
        # Preprocess image
        return image, transform(image).unsqueeze(0)

    def _predict(
        self,
        image_tensor,
    ):
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        #
        with torch.no_grad():

            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        return probabilities.cpu().numpy().flatten()

    def visualize_predictions(self, original_image, probabilities, class_names):
        fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

        axarr[0].imshow(original_image)
        axarr[0].set_title("Original image")

        axarr[1].barh(np.arange(len(class_names)), probabilities)
        axarr[1].set_yticks(np.arange(len(class_names)))
        axarr[1].set_yticklabels(class_names)
        axarr[1].invert_yaxis()
        axarr[1].set_title("Class probabilities")

        plt.show()
